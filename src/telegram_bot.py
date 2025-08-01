"""
Telegram Bot for CengBot - AI Assistant
=======================================

This module implements the main Telegram bot that handles user interactions,
processes messages, and communicates with the AI model through RabbitMQ.

Features:
- Telegram message handling
- Language detection (Turkish/English)
- Duplicate question detection
- User feedback system (like/dislike)
- RabbitMQ integration for AI processing
- SQLite database integration

Author: naholav
"""

import logging
import asyncio
import json
import threading
from datetime import datetime
from queue import Queue
from typing import Optional, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
from sqlalchemy.orm import Session
from sqlalchemy import text
from langdetect import detect, LangDetectException
import pika

# Import project modules
from database_models import SessionLocal, RawData, TrainingData, init_db
from config.env_loader import get_config

# Get configuration
config = get_config()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.log_level)
)
logger = logging.getLogger(__name__)

# Global queue for receiving answers from AI worker
answer_queue = Queue()


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code ('TR' for Turkish, 'EN' for English)
    """
    try:
        detected_lang = detect(text)
        return 'TR' if detected_lang == 'tr' else 'EN'
    except LangDetectException:
        logger.warning(f"Could not detect language for text: {text[:50]}...")
        return 'TR'  # Default to Turkish for Cukurova University


def check_similarity(db: Session, question: str) -> List[tuple]:
    """
    Check for similar questions in the database using simple text matching.
    
    This function performs a basic similarity check by looking for questions
    that contain similar text patterns. In production, this could be enhanced
    with other similarity algorithms.
    
    Args:
        db: Database session
        question: The question text to check
        
    Returns:
        List of similar questions as (id, question) tuples
    """
    try:
        # Simple similarity check using LIKE pattern matching
        # Take the first 20 characters as a pattern to find similar questions
        pattern = question[:20]
        
        results = db.execute(
            text("""
            SELECT id, question FROM raw_data 
            WHERE question LIKE :pattern
            AND id != (SELECT MAX(id) FROM raw_data)  -- Exclude the most recent (current) question
            LIMIT 5
            """),
            {"pattern": f"%{pattern}%"}
        ).fetchall()
        
        logger.info(f"Found {len(results)} similar questions for pattern: {pattern}")
        return results
        
    except Exception as e:
        logger.error(f"Error in similarity check: {e}")
        return []


def send_to_rabbitmq(message_data: dict) -> bool:
    """
    Send question data to RabbitMQ for AI processing.
    
    Args:
        message_data: Dictionary containing question data
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    try:
        # Establish RabbitMQ connection
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(config.rabbitmq_host, config.rabbitmq_port)
        )
        channel = connection.channel()
        
        # Declare the questions queue
        channel.queue_declare(queue=config.questions_queue, durable=True)
        
        # Send message to queue
        channel.basic_publish(
            exchange='',
            routing_key=config.questions_queue,
            body=json.dumps(message_data),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )
        
        connection.close()
        logger.info(f"✅ Sent question to RabbitMQ: ID {message_data.get('raw_data_id')}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send message to RabbitMQ: {e}")
        return False


def start_answer_consumer():
    """
    Start the RabbitMQ consumer for receiving AI-generated answers.
    
    This function runs in a separate thread to listen for answers from
    the AI worker and puts them in the global answer queue.
    """
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(config.rabbitmq_host, config.rabbitmq_port)
        )
        channel = connection.channel()
        
        # Declare the answers queue
        channel.queue_declare(queue=config.answers_queue, durable=True)
        
        def callback(ch, method, properties, body):
            """Process incoming answer from AI worker."""
            try:
                answer_data = json.loads(body)
                answer_queue.put(answer_data)
                logger.info(f"📥 Received answer from AI worker: ID {answer_data.get('raw_data_id')}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"❌ Error processing answer: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        channel.basic_consume(queue=config.answers_queue, on_message_callback=callback)
        logger.info("🔄 Started consuming answers from RabbitMQ")
        channel.start_consuming()
        
    except Exception as e:
        logger.error(f"❌ RabbitMQ answer consumer error: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming messages from Telegram users.
    
    This function processes user messages, detects language, checks for duplicates,
    saves to database, sends to AI worker via RabbitMQ, and responds to user.
    
    Args:
        update: Telegram update object containing message data
        context: Telegram context for bot operations
    """
    # Validate message
    if not update.message or not update.message.text:
        return
    
    # Check if message is in the correct topic (if specified)
    if config.telegram_topic_id and update.message.message_thread_id != config.telegram_topic_id:
        return
    
    # Extract user and message information
    user = update.effective_user
    message_text = update.message.text
    
    # Skip command messages (those starting with '/')
    if message_text.startswith('/'):
        return
    
    logger.info(f"📩 Received message from {user.username or user.first_name}: {message_text[:50]}...")
    
    # Initialize database session
    db = SessionLocal()
    try:
        # Detect language of the message
        detected_language = detect_language(message_text)
        logger.info(f"🌍 Detected language: {detected_language}")
        
        # Create new raw data entry in database
        raw_data = RawData(
            telegram_id=user.id,
            username=user.username or user.first_name,
            question=message_text,
            language=detected_language,
            message_thread_id=update.message.message_thread_id
        )
        db.add(raw_data)
        db.commit()
        db.refresh(raw_data)
        
        logger.info(f"💾 Saved question to database with ID: {raw_data.id}")
        
        # Check for duplicate questions
        similar_questions = check_similarity(db, message_text)
        if similar_questions and len(similar_questions) > 0:
            # Mark as duplicate and reference the first similar question
            raw_data.is_duplicate = True
            raw_data.duplicate_of_id = similar_questions[0][0]
            db.commit()
            logger.info(f"🔄 Marked as duplicate of question ID: {similar_questions[0][0]}")
        
        # Show typing indicator to user
        await update.message.chat.send_action("typing")
        
        # Prepare message data for RabbitMQ
        message_data = {
            'raw_data_id': raw_data.id,
            'telegram_id': user.id,
            'username': user.username or user.first_name,
            'question': message_text,
            'language': detected_language,
            'message_thread_id': update.message.message_thread_id,
            'update_message_id': update.message.message_id
        }
        
        # Send question to AI worker via RabbitMQ
        if not send_to_rabbitmq(message_data):
            # Fallback: respond with error message
            await update.message.reply_text(
                "⚠️ Sorry, I'm experiencing technical difficulties. Please try again later or contact the department office: +90 322 338 60 10"
            )
            return
        
        # Wait for AI response with timeout
        timeout_seconds = 30
        start_time = asyncio.get_event_loop().time()
        model_response = None
        
        while True:
            # Check for timeout
            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                model_response = "⏰ Response timeout. Please try again later or contact the department office for assistance."
                break
            
            # Check for response in queue
            if not answer_queue.empty():
                answer_data = answer_queue.get()
                if answer_data['raw_data_id'] == raw_data.id:
                    model_response = answer_data['answer']
                    break
                else:
                    # Put back if not for current question
                    answer_queue.put(answer_data)
            
            # Wait a bit before checking again
            await asyncio.sleep(0.5)
        
        # Update database with AI response
        raw_data.answer = model_response
        raw_data.answered_at = datetime.utcnow()
        db.commit()
        
        # Create inline keyboard for user feedback
        keyboard = [
            [
                InlineKeyboardButton("👍 Helpful", callback_data=f"like_{raw_data.id}"),
                InlineKeyboardButton("👎 Not Helpful", callback_data=f"dislike_{raw_data.id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Format response message
        response_text = (
            f"👤 **Question from @{user.username or user.first_name}:**\n"
            f"❓ {message_text}\n\n"
            f"🤖 **AI Assistant ({detected_language}):**\n"
            f"{model_response}\n\n"
            f"💡 Was this response helpful?"
        )
        
        # Send response to user
        sent_message = await update.message.reply_text(
            response_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Save Telegram message ID for future reference
        raw_data.telegram_message_id = sent_message.message_id
        db.commit()
        
        logger.info(f"✅ Sent response to user: {user.username or user.first_name}")
        
    except Exception as e:
        logger.error(f"❌ Error handling message: {e}")
        await update.message.reply_text(
            "❌ An error occurred. Please try again later."
        )
    finally:
        db.close()


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline button callbacks (like/dislike feedback).
    
    Args:
        update: Telegram update object containing callback data
        context: Telegram context for bot operations
    """
    query = update.callback_query
    await query.answer()
    
    try:
        # Parse callback data (format: "action_rawdataid")
        action, raw_data_id = query.data.split('_')
        raw_data_id = int(raw_data_id)
        
        db = SessionLocal()
        try:
            # Find the raw data record
            raw_data = db.query(RawData).filter(RawData.id == raw_data_id).first()
            if not raw_data:
                await query.answer("❌ Record not found!", show_alert=True)
                return
            
            # Update feedback based on action
            if action == "like":
                raw_data.like = 1  # Positive feedback
                await query.answer("👍 Thank you for your feedback!")
                logger.info(f"👍 Positive feedback from {query.from_user.username} for question ID: {raw_data_id}")
            elif action == "dislike":
                raw_data.like = -1  # Negative feedback
                await query.answer("👎 Thank you for your feedback. We'll improve!")
                logger.info(f"👎 Negative feedback from {query.from_user.username} for question ID: {raw_data_id}")
            else:
                await query.answer("❌ Invalid action!", show_alert=True)
                return
            
            db.commit()
            
            # Update the inline keyboard to show selected feedback
            keyboard = [
                [
                    InlineKeyboardButton(
                        "👍 Helpful" + (" ✓" if raw_data.like == 1 else ""), 
                        callback_data=f"like_{raw_data_id}"
                    ),
                    InlineKeyboardButton(
                        "👎 Not Helpful" + (" ✓" if raw_data.like == -1 else ""), 
                        callback_data=f"dislike_{raw_data_id}"
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Update the message with new keyboard
            await query.edit_message_reply_markup(reply_markup=reply_markup)
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"❌ Error handling callback: {e}")
        await query.answer("❌ An error occurred!", show_alert=True)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command - welcome message and bot information.
    
    Args:
        update: Telegram update object
        context: Telegram context for bot operations
    """
    welcome_message = (
        "🤖 **Çukurova University Computer Engineering AI Assistant**\n\n"
        "🧠 **AI Engine:** LLaMA 3.2 3B + LoRA Fine-tuned\n"
        "💾 **Database:** SQLite\n"
        "🔥 **Status:** Production Ready\n\n"
        "📝 Hello! I'm here to answer your questions about Computer Engineering department.\n\n"
        "💡 **Example questions:**\n"
        "• What courses are available?\n"
        "• How to do internship?\n"
        "• What are graduation requirements?\n"
        "• Is there an Erasmus program?\n\n"
        "🚀 **Feel free to ask your questions!**"
    )
    
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN
    )
    logger.info(f"🚀 /start command from: {update.effective_user.username or update.effective_user.first_name}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /stats command - show bot statistics.
    
    Args:
        update: Telegram update object
        context: Telegram context for bot operations
    """
    db = SessionLocal()
    try:
        # Gather statistics from database
        total_questions = db.query(RawData).count()
        liked_questions = db.query(RawData).filter(RawData.like == 1).count()
        disliked_questions = db.query(RawData).filter(RawData.like == -1).count()
        approved_questions = db.query(RawData).filter(RawData.admin_approved == 1).count()
        training_data_count = db.query(TrainingData).count()
        duplicate_count = db.query(RawData).filter(RawData.is_duplicate == True).count()
        
        # Calculate success rate
        total_feedback = liked_questions + disliked_questions
        success_rate = "N/A"
        if total_feedback > 0:
            success_rate = f"{(liked_questions / total_feedback * 100):.1f}%"
        
        # Format statistics message
        stats_text = f"""📊 **AI Assistant Statistics**

🤖 **Model:** LLaMA 3.2 3B + LoRA
📝 **Total Questions:** {total_questions}
👍 **Liked Responses:** {liked_questions}
👎 **Disliked Responses:** {disliked_questions}
✅ **Approved by Admin:** {approved_questions}
🎓 **Training Data:** {training_data_count}
🔄 **Duplicate Questions:** {duplicate_count}
📈 **Success Rate:** {success_rate}

🔥 **Status:** Active and Ready
        """
        
        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"📊 /stats command from: {update.effective_user.username or update.effective_user.first_name}")
        
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        await update.message.reply_text("❌ Could not retrieve statistics.")
    finally:
        db.close()


def main() -> None:
    """
    Main function to start the Telegram bot.
    
    Initializes the database, starts the RabbitMQ consumer, and runs the bot.
    """
    logger.info("🚀 Starting CengBot Telegram Bot...")
    
    # Initialize database
    init_db()
    logger.info("✅ Database initialized")
    
    # Start RabbitMQ answer consumer in background thread
    consumer_thread = threading.Thread(target=start_answer_consumer, daemon=True)
    consumer_thread.start()
    logger.info("✅ RabbitMQ answer consumer started")
    
    # Create Telegram bot application
    application = Application.builder().token(config.telegram_bot_token).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))
    
    # Add message handler for regular text messages
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        handle_message
    ))
    
    # Add callback handler for inline button responses
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Start the bot
    logger.info("🚀 Telegram bot is ready and running in production!")
    logger.info("📱 You can now test the bot on Telegram!")
    
    # Run the bot until interrupted
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()