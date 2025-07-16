import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from database_models import SessionLocal, RawData, TrainingData, init_db
import asyncio
import pika
import json
import threading
from queue import Queue
from langdetect import detect

# Bot configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_TOPIC_ID = None  # Will be set to specific topic ID if needed

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global queue for answers
answer_queue = Queue()

def check_similarity(db: Session, question: str) -> list:
    """Check for similar questions - simplified for SQLite"""
    try:
        # Simple similarity check for SQLite
        results = db.execute(
            text("""
            SELECT id, question FROM raw_data 
            WHERE question LIKE :pattern
            LIMIT 5
            """),
            {"pattern": f"%{question[:20]}%"}
        ).fetchall()
        return results
    except Exception as e:
        logger.error(f"Similarity check error: {e}")
        return []

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages in group"""
    if not update.message or not update.message.text:
        return
    
    # Check if message is in correct topic (if BOT_TOPIC_ID is set)
    if BOT_TOPIC_ID and update.message.message_thread_id != BOT_TOPIC_ID:
        return
    
    # Get user info
    user = update.effective_user
    message_text = update.message.text
    
    # Skip if message is a command
    if message_text.startswith('/'):
        return
    
    db = SessionLocal()
    try:
        # Detect language
        try:
            detected_lang = detect(message_text)
            language = 'TR' if detected_lang == 'tr' else 'EN'
        except:
            language = 'TR'  # Default to Turkish
        
        # Save question to database
        raw_data = RawData(
            telegram_id=user.id,
            username=user.username or user.first_name,
            question=message_text,
            language=language,
            message_thread_id=update.message.message_thread_id
        )
        db.add(raw_data)
        db.commit()
        db.refresh(raw_data)
        
        # Check for duplicates
        similar_questions = check_similarity(db, message_text)
        if similar_questions and len(similar_questions) > 1:  # Exclude self
            raw_data.is_duplicate = True
            raw_data.duplicate_of_id = similar_questions[1][0]  # First similar (excluding self)
            db.commit()
        
        # Send typing action
        await update.message.chat.send_action("typing")
        
        # Send to RabbitMQ instead of direct model call
        send_to_rabbitmq({
            'raw_data_id': raw_data.id,
            'telegram_id': user.id,
            'username': user.username or user.first_name,
            'question': message_text,
            'message_thread_id': update.message.message_thread_id,
            'update_message_id': update.message.message_id
        })
        
        # Wait for response (with timeout)
        timeout = 30  # seconds
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if not answer_queue.empty():
                answer_data = answer_queue.get()
                if answer_data['raw_data_id'] == raw_data.id:
                    model_response = answer_data['answer']
                    break
                else:
                    # Put it back if not for us
                    answer_queue.put(answer_data)
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                model_response = "Sorry, I cannot generate a response at the moment. Please try again later."
                break
            
            await asyncio.sleep(0.1)
        
        # Create inline keyboard for feedback
        keyboard = [
            [
                InlineKeyboardButton("👍", callback_data=f"like_{raw_data.id}"),
                InlineKeyboardButton("👎", callback_data=f"dislike_{raw_data.id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send response with mention
        response_text = f"@{user.username or user.first_name} asked:\n❓ {message_text}\n\n🤖 Answer:\n{model_response}"
        
        # Send message and save message_id
        sent_message = await update.message.reply_text(
            response_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
        
        # Update telegram_message_id
        raw_data.telegram_message_id = sent_message.message_id
        db.commit()
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text("An error occurred. Please try again later.")
    finally:
        db.close()

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    # Parse callback data
    action, raw_data_id = query.data.split('_')
    raw_data_id = int(raw_data_id)
    
    db = SessionLocal()
    try:
        # Get the record
        raw_data = db.query(RawData).filter(RawData.id == raw_data_id).first()
        if not raw_data:
            await query.answer("Record not found!", show_alert=True)
            return
        
        # Update point status (1 for like, -1 for dislike)
        if action == "like":
            raw_data.like = 1
            await query.answer("You liked it! 👍")
        else:
            raw_data.like = -1
            await query.answer("You disliked it! 👎")
        
        db.commit()
        
        # Update button text to show current status
        keyboard = [
            [
                InlineKeyboardButton(
                    "👍" + (" ✓" if raw_data.like == 1 else ""), 
                    callback_data=f"like_{raw_data_id}"
                ),
                InlineKeyboardButton(
                    "👎" + (" ✓" if raw_data.like == -1 else ""), 
                    callback_data=f"dislike_{raw_data_id}"
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_reply_markup(reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error handling callback: {e}")
        await query.answer("An error occurred!", show_alert=True)
    finally:
        db.close()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    await update.message.reply_text(
        "Hello! I am the Çukurova University Computer Engineering AI assistant.\n"
        "You can write your questions to the group, and I will try to help you."
    )

def main() -> None:
    """Start the bot"""
    # Initialize database
    init_db()
    
    # Model is now loaded in RabbitMQ worker
    logger.info("Bot starting... Model will be loaded in RabbitMQ worker.")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS, 
        handle_message
    ))
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Start polling
    logger.info("Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

def send_to_rabbitmq(data):
    """Send question to RabbitMQ queue"""
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='questions', durable=True)
        
        channel.basic_publish(
            exchange='',
            routing_key='questions',
            body=json.dumps(data),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )
        connection.close()
        logger.info(f"Sent question to RabbitMQ: {data['raw_data_id']}")
    except Exception as e:
        logger.error(f"RabbitMQ send error: {e}")

def consume_answers():
    """Consume answers from RabbitMQ in separate thread"""
    def callback(ch, method, properties, body):
        try:
            data = json.loads(body)
            answer_queue.put(data)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Received answer for question ID: {data['raw_data_id']}")
        except Exception as e:
            logger.error(f"Error processing answer: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            channel = connection.channel()
            channel.queue_declare(queue='answers', durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='answers', on_message_callback=callback)
            
            logger.info("Started consuming answers from RabbitMQ")
            channel.start_consuming()
        except Exception as e:
            logger.error(f"RabbitMQ consumer error: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Start answer consumer in separate thread
    import time
    consumer_thread = threading.Thread(target=consume_answers, daemon=True)
    consumer_thread.start()
    
    # Start bot
    main()