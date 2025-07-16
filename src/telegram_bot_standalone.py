import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from database_models import SessionLocal, RawData, TrainingData, init_db
from llama_model_handler import model_instance
import asyncio
import os

# Bot configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
    """Handle incoming messages"""
    if not update.message or not update.message.text:
        return
    
    user = update.effective_user
    message_text = update.message.text
    
    if message_text.startswith('/'):
        return
    
    logger.info(f"📩 {user.username or user.first_name}: {message_text}")
    
    db = SessionLocal()
    try:
        # Save question
        raw_data = RawData(
            telegram_id=user.id,
            username=user.username or user.first_name,
            question=message_text,
            message_thread_id=update.message.message_thread_id
        )
        db.add(raw_data)
        db.commit()
        db.refresh(raw_data)
        
        # Check for duplicates
        similar_questions = check_similarity(db, message_text)
        if similar_questions and len(similar_questions) > 1:
            raw_data.is_duplicate = True
            raw_data.duplicate_of_id = similar_questions[1][0]
            db.commit()
        
        await update.message.chat.send_action("typing")
        
        # Get model response
        try:
            logger.info("🤖 Generating response with LLaMA 3.2 3B + LoRA...")
            model_response = model_instance.generate_response(message_text)
            logger.info("✅ AI response generated")
        except Exception as e:
            logger.error(f"❌ Model error: {e}")
            model_response = "I apologize, I'm experiencing a technical issue at the moment. Please try again later or contact the department secretary: +90 322 338 60 10"
        
        # Update database
        raw_data.answer = model_response
        raw_data.answered_at = datetime.utcnow()
        db.commit()
        
        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("👍", callback_data=f"like_{raw_data.id}"),
                InlineKeyboardButton("👎", callback_data=f"dislike_{raw_data.id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send response
        response_text = f"👤 @{user.username or user.first_name} asked:\n❓ {message_text}\n\n🤖 AI Answer:\n{model_response}"
        
        sent_message = await update.message.reply_text(
            response_text,
            reply_markup=reply_markup
        )
        
        raw_data.telegram_message_id = sent_message.message_id
        db.commit()
        
        logger.info(f"✅ Response sent: {user.username or user.first_name}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await update.message.reply_text("An error occurred. Please try again later.")
    finally:
        db.close()

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    action, raw_data_id = query.data.split('_')
    raw_data_id = int(raw_data_id)
    
    db = SessionLocal()
    try:
        raw_data = db.query(RawData).filter(RawData.id == raw_data_id).first()
        if not raw_data:
            await query.answer("Record not found!", show_alert=True)
            return
        
        if action == "like":
            raw_data.like = 1
            await query.answer("You liked it! 👍")
            logger.info(f"👍 {query.from_user.username} - Q{raw_data_id}")
        else:
            raw_data.like = -1
            await query.answer("You disliked it! 👎")
            logger.info(f"👎 {query.from_user.username} - Q{raw_data_id}")
        
        db.commit()
        
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
        logger.error(f"❌ Callback error: {e}")
        await query.answer("An error occurred!", show_alert=True)
    finally:
        db.close()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 **Çukurova University Computer Engineering AI Assistant**\n\n"
        "🧠 **AI Engine:** LLaMA 3.2 3B + LoRA Fine-tuned\n"
        "💾 **Database:** SQLite\n"
        "🔥 **Status:** Production Ready\n\n"
        "📝 Hello! I'm here to answer your questions about Computer Engineering.\n\n"
        "💡 **Example questions:**\n"
        "• What courses are available?\n"
        "• How to do internship?\n"
        "• What are the graduation requirements?\n"
        "• Is there an Erasmus program?\n\n"
        "🚀 **You can ask your questions!**",
        parse_mode=ParseMode.MARKDOWN
    )
    logger.info(f"🚀 Start: {update.effective_user.username or update.effective_user.first_name}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command"""
    db = SessionLocal()
    try:
        total = db.query(RawData).count()
        liked = db.query(RawData).filter(RawData.like == 1).count()
        disliked = db.query(RawData).filter(RawData.like == -1).count()
        approved = db.query(RawData).filter(RawData.admin_approved == 1).count()
        training_count = db.query(TrainingData).count()
        
        success_rate = "N/A"
        if (liked + disliked) > 0:
            success_rate = f"{(liked/(liked+disliked)*100):.1f}%"
            
        stats_text = f"""📊 **AI Assistant Statistics**

🤖 **Model:** LLaMA 3.2 3B + LoRA
📝 **Total Questions:** {total}
👍 **Liked:** {liked}
👎 **Disliked:** {disliked}
✅ **Approved:** {approved}
🎓 **Training Data:** {training_count}
📈 **Success Rate:** {success_rate}

🔥 **Status:** Active and Ready
        """
        
        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"📊 Stats: {update.effective_user.username or update.effective_user.first_name}")
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        await update.message.reply_text("Could not retrieve statistics.")
    finally:
        db.close()

def main() -> None:
    """Start the bot"""
    logger.info("🚀 Starting University Bot Final Production...")
    
    # Initialize database
    init_db()
    logger.info("✅ SQLite Database ready")
    
    # Load model
    logger.info("🤖 Loading LLaMA 3.2 3B + LoRA model...")
    success = model_instance.load_model()
    if not success:
        logger.error("❌ Model could not be loaded! Stopping bot.")
        return
    
    logger.info("🎉 AI Model successfully loaded!")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        handle_message
    ))
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Start polling
    logger.info("🚀 Bot ready and running in production!")
    logger.info("📱 You can test it on Telegram!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()