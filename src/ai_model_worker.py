import pika
import json
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from sqlalchemy.orm import Session
from database_models import SessionLocal, RawData
from llama_model_handler import model_instance
from config.env_loader import load_config
import time

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RabbitMQWorker:
    def __init__(self):
        self.config = load_config()
        self.connection = None
        self.channel = None
        self.connect()
        
    def connect(self):
        """Connect to RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.config.rabbitmq_host,
                    port=self.config.rabbitmq_port
                )
            )
            self.channel = self.connection.channel()
            
            # Declare queues
            self.channel.queue_declare(queue=self.config.questions_queue, durable=True)
            self.channel.queue_declare(queue=self.config.answers_queue, durable=True)
            
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"RabbitMQ connection error: {e}")
            raise
    
    def process_question(self, ch, method, properties, body):
        """Process incoming question"""
        try:
            # Parse message
            data = json.loads(body)
            logger.info(f"🔄 RabbitMQ: Processing question: {data['question'][:50]}...")
            
            db = SessionLocal()
            try:
                # Get the raw data record with retry mechanism
                raw_data = None
                for attempt in range(5):  # Try up to 5 times
                    raw_data = db.query(RawData).filter(RawData.id == data['raw_data_id']).first()
                    if raw_data:
                        break
                    logger.warning(f"RawData with id {data['raw_data_id']} not found, attempt {attempt + 1}/5")
                    time.sleep(0.1)  # Wait 100ms before retry
                    
                if not raw_data:
                    logger.error(f"RawData with id {data['raw_data_id']} not found after 5 attempts")
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    return
                
                # Generate response
                response = model_instance.generate_response(data['question'])
                
                # Update database
                raw_data.answer = response
                raw_data.answered_at = datetime.utcnow()
                db.commit()
                
                # Send to answers queue
                answer_data = {
                    'raw_data_id': data['raw_data_id'],
                    'telegram_id': data['telegram_id'],
                    'message_thread_id': data.get('message_thread_id'),
                    'username': data['username'],
                    'question': data['question'],
                    'answer': response,
                    'update_message_id': data.get('update_message_id')
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key='answers',
                    body=json.dumps(answer_data),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    )
                )
                
                logger.info(f"✅ RabbitMQ: Answer sent for question ID: {data['raw_data_id']}")
                
            finally:
                db.close()
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"❌ RabbitMQ: Error processing question: {e}")
            # Reject and requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start_consuming(self):
        """Start consuming messages"""
        logger.info("Loading model...")
        success = model_instance.load_model()
        if not success:
            logger.error("Failed to load model!")
            return
        
        logger.info("Model loaded successfully. Starting to consume messages...")
        
        # Set up consumer
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue='questions',
            on_message_callback=self.process_question
        )
        
        try:
            logger.info("Waiting for messages...")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.channel.stop_consuming()
            self.connection.close()

def main():
    while True:
        try:
            worker = RabbitMQWorker()
            worker.start_consuming()
        except Exception as e:
            logger.error(f"Worker crashed: {e}")
            logger.info("Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == '__main__':
    main()