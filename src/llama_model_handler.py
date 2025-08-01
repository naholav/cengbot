import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel
import logging
from langdetect import detect, LangDetectException
try:
    from .error_handler import handle_model_error, handle_error, ErrorLevel
except ImportError:
    from error_handler import handle_model_error, handle_error, ErrorLevel

logger = logging.getLogger(__name__)

class ModelConfig:
    # Model paths - Updated for active model system
    base_model = "meta-llama/Llama-3.2-3B"
    lora_model_path = "models/active-model"  # Uses active-model symlink (method1 added dynamically if exists)
    cache_dir = "model_cache"
    use_local_cache = True
    
    # Generation parameters - AYNEN KORUNDU
    temperature = 0.7
    max_new_tokens = 200
    repetition_penalty = 1.1
    top_p = 0.95
    top_k = 50
    
    # System prompts 
    system_prompt_tr = """Sen Çukurova Üniversitesi Bilgisayar Mühendisliği bölümünün deneyimli dijital asistanısın. Öğrencilere samimi, yardımsever ve doğru bilgiler vererek destek oluyorsun.

Önemli kurallar:
- Her soruya MAKSİMUM 4 CÜMLE ile yanıt ver
- Cevapların çok kısa, net ve anlaşılır olmalı
- Gereksiz açıklamalardan ve tekrarlardan kesinlikle kaçın
- Selamlama mesajlarına tek cümlelik karşılık ver
- Sadece sorulan soruya odaklan, ekstra bilgi verme
- Cevaplarında MAKSİMUM 1 adet soru sorabilirsin
- Karşı tarafa çok fazla soru sorma, sadece gerektiğinde tek soru sor"""
    
    system_prompt_en = """You are an experienced digital assistant for Çukurova University Computer Engineering Department. You help students by providing friendly, helpful and accurate information.

Important rules:
- Answer each question with MAXIMUM 4 SENTENCES
- Keep your answers very short, clear and understandable
- Absolutely avoid unnecessary explanations and repetitions
- Reply to greeting messages with a single sentence
- Focus only on the asked question, don't provide extra information
- You can ask MAXIMUM 1 question in your answers
- Don't ask too many questions, only ask one when necessary"""
    
    # HuggingFace token
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

class CengBotModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig()
        logger.info(f"Using device: {self.device}")
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            clean_text = text.strip().lower()
            
            # Quick check for common Turkish characters
            turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü']
            if any(char in clean_text for char in turkish_chars):
                return 'tr'
            
            detected_lang = detect(clean_text)
            
            if detected_lang == 'tr':
                return 'tr'
            else:
                return 'en'
                
        except LangDetectException:
            turkish_words = ['merhaba', 'selam', 'nedir', 'nasıl', 'hangi', 'var', 'mı', 'mi', 'mu', 'mü']
            if any(word in text.lower() for word in turkish_words):
                return 'tr'
            return 'en'
    
    def load_model(self):
        """Load the base model and LoRA weights"""
        logger.info("Loading model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                token=self.config.hf_token,
                trust_remote_code=True,
                use_fast=True,
                cache_dir=self.config.cache_dir
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            # Load base model
            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.config.hf_token,
                trust_remote_code=True,
                use_cache=True,
                cache_dir=self.config.cache_dir,
                force_download=False
            )
            
            # Load LoRA weights - check for both v1 style (method1 subdir) and v1.1 style (direct)
            lora_path = self.config.lora_model_path
            method1_path = os.path.join(lora_path, "method1")
            
            # Determine the correct path to use
            if os.path.exists(method1_path) and os.path.exists(os.path.join(method1_path, "adapter_model.safetensors")):
                # v1 style with method1 subdirectory
                actual_lora_path = method1_path
                logger.info(f"Found v1 style model structure (with method1 subdirectory)")
            elif os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
                # v1.1 style with direct adapter files
                actual_lora_path = lora_path
                logger.info(f"Found v1.1 style model structure (direct adapter files)")
            else:
                logger.error(f"❌ LoRA adapter not found at {lora_path} or {method1_path}")
                raise FileNotFoundError(f"LoRA adapter required but not found")
            
            logger.info(f"Loading LoRA weights from {actual_lora_path}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                actual_lora_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info("✅ LoRA adapter loaded successfully!")
            
            self.model.eval()
            logger.info("🎉 LLaMA 3.2 3B + LoRA model loaded successfully!")
            return True
            
        except Exception as e:
            error_response = handle_model_error(e, "load_model")
            logger.error(f"Failed to load model: {e}")
            return False
    
    def format_prompt(self, message: str) -> tuple:
        """Format the prompt with system prompt based on detected language"""
        lang = self.detect_language(message)
        logger.info(f"Detected language: {lang}")
        
        if lang == 'tr':
            system_prompt = self.config.system_prompt_tr
            formatted_prompt = f"{system_prompt}\n\nStudent: {message}\nAssistant:"
        else:
            system_prompt = self.config.system_prompt_en
            formatted_prompt = f"{system_prompt}\n\nStudent: {message}\nAssistant:"
        
        return formatted_prompt, lang
    
    def generate_response(self, message: str):
        """Generate response from the model"""
        if self.model is None:
            return "Model not loaded."
        
        try:
            # Format prompt and get language
            prompt, lang = self.format_prompt(message)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Check if it's a greeting
            greetings_tr = ["selam", "merhaba", "hey", "slm", "mrb", "günaydın", "iyi günler", "iyi akşamlar"]
            greetings_en = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
            
            is_greeting = any(greeting in message.lower() for greeting in (greetings_tr + greetings_en))
            
            # Adjust max tokens for greetings
            max_tokens = 30 if is_greeting else 100
            
            # Create generation config
            generation_config = GenerationConfig(
                temperature=self.config.temperature,
                max_new_tokens=max_tokens,
                repetition_penalty=self.config.repetition_penalty,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if lang == 'tr':
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                    if "Student:" in response:
                        response = response.split("Student:")[0].strip()
            else:
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                    if "Student:" in response:
                        response = response.split("Student:")[0].strip()
            
            # Limit to 3 sentences
            sentences = response.split('. ')
            if len(sentences) > 3:
                sentences = sentences[:3]
            response = '. '.join(sentences)
            
            # Count question marks and limit to 1
            question_count = response.count('?')
            if question_count > 1:
                parts = response.split('?')
                response = parts[0] + '?'
                for part in parts[1:]:
                    if part.strip() and '?' not in part:
                        if lang == 'tr':
                            if not any(q in part.lower() for q in ['mı', 'mi', 'mu', 'mü']):
                                response += ' ' + part.strip() + '.'
                                break
                        else:
                            response += ' ' + part.strip() + '.'
                            break
            
            # Ensure it ends with proper punctuation
            if response and not response.endswith(('.', '!', '?')):
                response += '.'
            
            return response
            
        except Exception as e:
            error_response = handle_model_error(e, "generate_response")
            logger.error(f"Generation error: {e}")
            return "Sorry, I encountered an error while generating a response. Please try again."

# Global model instance
model_instance = CengBotModel()