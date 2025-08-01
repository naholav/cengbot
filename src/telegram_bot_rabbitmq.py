import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from database_models import SessionLocal, RawData, TrainingData, init_db, mark_duplicate_questions, handle_user_vote, get_vote_statistics
import asyncio
import pika
import json
import threading
from queue import Queue
from langdetect import detect
from collections import defaultdict
import time
from config.env_loader import load_config

# Load configuration
config = load_config()

# Bot configuration
BOT_TOKEN = config.telegram_bot_token
BOT_TOPIC_ID = config.telegram_topic_id

# SECURITY: Only allow this specific group
ALLOWED_CHAT_ID = -1002630398173  # The specific group ID from the URL

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global queue for answers
answer_queue = Queue()

# Rate limiting configuration
RATE_LIMIT_MESSAGES = 5  # Messages per minute
RATE_LIMIT_WINDOW = 60  # Time window in seconds
user_message_history = defaultdict(list)

def is_rate_limited(user_id: int) -> bool:
    """Check if user has exceeded rate limit"""
    current_time = time.time()
    
    # Clean old entries
    user_message_history[user_id] = [
        timestamp for timestamp in user_message_history[user_id]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if limit exceeded
    if len(user_message_history[user_id]) >= RATE_LIMIT_MESSAGES:
        return True
    
    # Add current message timestamp
    user_message_history[user_id].append(current_time)
    return False


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages in group"""
    if not update.message or not update.message.text:
        return
    
    # SECURITY: Only process messages from the allowed group
    if update.message.chat.id != ALLOWED_CHAT_ID:
        logger.warning(f"Unauthorized access attempt from chat ID: {update.message.chat.id}")
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
    
    # Check rate limiting
    if is_rate_limited(user.id):
        await update.message.reply_text(
            "⚠️ You're sending messages too quickly. Please wait a moment before sending another message.",
            reply_to_message_id=update.message.message_id
        )
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
        
        # Automatically check for duplicates using cosine similarity
        mark_duplicate_questions(db, raw_data.id, raw_data.question)
        
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
        
        # Create inline keyboard for feedback with initial vote counts
        vote_stats = get_vote_statistics(db, raw_data.id)
        keyboard = [
            [
                InlineKeyboardButton(f"👍 {vote_stats['likes']}", callback_data=f"like_{raw_data.id}"),
                InlineKeyboardButton(f"👎 {vote_stats['dislikes']}", callback_data=f"dislike_{raw_data.id}")
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
        
        # Update telegram_message_id with bot's response message ID
        raw_data.telegram_message_id = sent_message.message_id
        db.commit()
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text("An error occurred. Please try again later.")
    finally:
        db.close()

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks with user-based voting system"""
    query = update.callback_query
    
    # SECURITY: Only process callbacks from the allowed group
    if query.message and query.message.chat.id != ALLOWED_CHAT_ID:
        logger.warning(f"Unauthorized callback attempt from chat ID: {query.message.chat.id}")
        await query.answer("Unauthorized access!", show_alert=True)
        return
    
    # Parse callback data
    action, raw_data_id = query.data.split('_')
    raw_data_id = int(raw_data_id)
    user_id = query.from_user.id
    
    db = SessionLocal()
    try:
        # Get the record
        raw_data = db.query(RawData).filter(RawData.id == raw_data_id).first()
        if not raw_data:
            await query.answer("Record not found!", show_alert=True)
            return
        
        # Handle user vote
        vote_type = 1 if action == "like" else -1
        vote_result = handle_user_vote(db, raw_data_id, user_id, vote_type)
        
        # Answer with vote result message
        await query.answer(vote_result["message"], show_alert=not vote_result["success"])
        
        if vote_result["success"]:
            # Get updated vote statistics
            vote_stats = get_vote_statistics(db, raw_data_id)
            
            # Update button text to show vote counts
            keyboard = [
                [
                    InlineKeyboardButton(
                        f"👍 {vote_stats['likes']}", 
                        callback_data=f"like_{raw_data_id}"
                    ),
                    InlineKeyboardButton(
                        f"👎 {vote_stats['dislikes']}", 
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
    # SECURITY: Only respond to start command in the allowed group
    if update.message.chat.id != ALLOWED_CHAT_ID:
        logger.warning(f"Unauthorized start command from chat ID: {update.message.chat.id}")
        return
    
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
    logger.info(f"SECURITY: Bot configured to ONLY work in group: {ALLOWED_CHAT_ID}")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers - ONLY for the specific group
    application.add_handler(CommandHandler("start", start_command, filters=filters.Chat(ALLOWED_CHAT_ID)))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.Chat(ALLOWED_CHAT_ID) & filters.ChatType.GROUPS, 
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