#!/usr/bin/env python3
"""
CengBot Data Augmentation System
================================

This module uses Anthropic's Claude API to augment training data from the database.
It generates multiple variations of existing question-answer pairs to improve
model training diversity and performance.

Features:
- Database-driven data augmentation
- High-quality prompt engineering for Turkish and English
- Error handling and retry logic
- Progress tracking and checkpoint system
- Link preservation and validation
- Automatic export to training format

Security:
- API key is requested from user (not stored anywhere)
- No hardcoded credentials
- Secure API key handling

Author: naholav
"""

import json
import os
import sys
import time
import re
import shutil
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database models
from src.database_models import SessionLocal, TrainingData

# Anthropic API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class CengBotDataAugmenter:
    """Augment training data using Anthropic Claude API"""
    
    def __init__(self, api_key: str, base_path: str = "/home/ceng/cu_ceng_bot"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Please install with: pip install anthropic")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        self.base_path = Path(base_path)
        self.data_dir = self.base_path / "data"
        self.output_file = self.data_dir / "cengbot_qa_augmented.jsonl"
        self.checkpoint_file = self.data_dir / "augment_checkpoint.json"
        
        # API settings
        self.model_name = "claude-3-sonnet-20240229"  # Default model
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_retries = 3
        self.retry_delay = 2
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        
        # Turkish prompt for augmentation
        self.turkish_prompt = """Sen yıllardır doğal dil işleme ve veri zenginleştirme alanında uzmanlaşmış bir AI mühendisisin. Üniversite öğrencilerinin chatbot'larla nasıl konuştuğunu çok iyi biliyorsun.

ÖNEMLİ: Bu varyasyonlar Llama 3.2 3B modelini fine-tune etmek için kullanılacak. Varyasyonların kalitesi, çeşitliliği ve doğallığı modelin performansını doğrudan etkileyecek.

MUTLAK KURALLAR:
1. Link, isim, sayı, tarih ASLA değişmez
2. Yeni bilgi EKLEME
3. Gerçek öğrenciler gibi yaz
4. Adım adım anlatırken ASLA rakam kullanma (1,2,3 gibi)
5. Ders kodları, hoca isimleri, bölüm isimleri AYNEN korunmalı
6. Her varyasyon anlamsal olarak orijinalle %100 uyumlu olmalı
7. YAZIM HATASI KURALI: Sadece "8. Yazım hatalı" stilini kullanırken yazım hatası yap. Diğer stillerde ASLA yazım hatası yapma.
8. HİTAP KURALI: ASLA "Değerli hocam", "Sayın hocam", "Muhterem", "Kıymetli" gibi aşırı resmi hitaplar kullanma.
9. DOĞAL DİL KURALI: Öğrenciler chatbot'a yazarken doğal konuşurlar. "Merhaba", "Selam", doğrudan soru sorma veya hiç hitap kullanmama tercih edilmeli.

SORU STİLLERİ:
1. Ultra kısa 
2. Günlük dil 
3. Nazik ama doğal
4. Problem anlatarak
5. Çoklu soru 
6. Deneyim paylaşarak
7. İngilizce terimli
8. Yazım hatalı 
9. Bağlam vererek soru
10. Dolaylı/ima ederek
11. Acil/panik modda
12. Spesifik detay isteyerek
13. Kısaltmalar kullanarak
14. Belirsizlik ifadesiyle
15. Yarı resmi

CEVAP STİLLERİ:
1. Tek cümle net bilgi
2. Samimi destek
3. Detaylı açıklama
4. Öneri şeklinde
5. Adım adım 
6. Karşılaştırma
7. Teknik + ekstra
8. FAQ tarzı
9. Deneyim paylaşımlı
10. Motive edici/cesaretlendirici
11. Uyarılar ve dikkat edileceklerle
12. Sebep-sonuç ilişkisiyle
13. Resmi ama anlaşılır
14. Özet + detay kombinasyonu
15. Konuşma dili ağırlıklı

ÖĞRENCİ PROFİLLERİ (Varyasyonlarda kullan):
- Hazırlık öğrencisi (kafası karışık)
- 1. sınıf (her şey yeni)
- Son sınıf (mezuniyet telaşı)
- Yüksek lisans öğrencisi (detaycı)
- Çift anadal yapan (yoğun)
- Online öğrenci (uzaktan)

Soru: {question}
Cevap: {answer}

ÇIKTI TALİMATI:
15 varyasyon üret. Her biri:
1. Farklı bir stil kombinasyonu kullansın
2. Farklı bir öğrenci profili yansıtsın
3. Çeşitlilik skoru > 0.6 olsun
4. Doğallık testi: Gerçek bir öğrenci böyle yazar mıydı?

JSON array formatında döndür. Her varyasyonda 'question' ve 'answer' alanları olsun."""

        # English prompt for augmentation
        self.english_prompt = """You are an AI engineer with years of expertise in natural language processing and data augmentation. You know very well how university students interact with chatbots.

IMPORTANT: These variations will be used to fine-tune the Llama 3.2 3B model. The quality, diversity, and naturalness of the variations will directly affect the model's performance.

ABSOLUTE RULES:
1. Links, names, numbers, dates NEVER change
2. DO NOT add new information
3. Write like real students
4. When explaining step by step, NEVER use numbers (1,2,3 etc.)
5. Course codes, professor names, department names must remain EXACTLY the same
6. Each variation must be 100% semantically aligned with the original
7. TYPO RULE: Only make typos when using "8. With typos" style. NEVER make typos in other styles.
8. GREETING RULE: NEVER use overly formal greetings like "Dear professor", "Esteemed teacher", "Respected sir/madam".
9. NATURAL LANGUAGE RULE: Students write naturally to chatbots. Use "Hi", "Hello", direct questions, or no greeting at all.

QUESTION STYLES:
1. Ultra short 
2. Casual language 
3. Polite but natural
4. Explaining a problem
5. Multiple questions 
6. Sharing experience
7. Mixed technical terms
8. With typos 
9. Providing context
10. Indirect/implying
11. Urgent/panic mode
12. Asking for specific details
13. Using abbreviations
14. With uncertainty expressions
15. Semi-formal

ANSWER STYLES:
1. Single sentence direct info
2. Friendly support
3. Detailed explanation
4. Suggestion format
5. Step by step 
6. Comparison
7. Technical + extra
8. FAQ style
9. Experience sharing
10. Motivating/encouraging
11. With warnings and cautions
12. Cause-effect relationship
13. Formal but understandable
14. Summary + detail combination
15. Conversational heavy

STUDENT PROFILES (Use in variations):
- Prep student (confused)
- Freshman (everything is new)
- Senior (graduation rush)
- Graduate student (detail-oriented)
- Double major student (busy)
- Online student (remote)

Question: {question}
Answer: {answer}

OUTPUT INSTRUCTION:
Generate 15 variations. Each should:
1. Use a different style combination
2. Reflect a different student profile
3. Have diversity score > 0.6
4. Naturalness test: Would a real student write like this?

Return as JSON array. Each variation should have 'question' and 'answer' fields."""

    def load_checkpoint(self) -> Dict:
        """Load checkpoint data"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"last_processed_index": -1, "output_data": []}
    
    def save_checkpoint(self, index: int, output_data: List[Dict]):
        """Save checkpoint data"""
        checkpoint = {
            "last_processed_index": index,
            "timestamp": datetime.now().isoformat(),
            "output_data": output_data
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def load_training_data_from_db(self) -> List[Dict]:
        """Load approved training data from database"""
        db = SessionLocal()
        training_data = []
        
        try:
            # Query active training data
            records = db.query(TrainingData).filter(
                TrainingData.is_active == True
            ).order_by(TrainingData.created_at).all()
            
            print(f"Found {len(records)} approved training records in database")
            
            for record in records:
                training_data.append({
                    "id": record.id,
                    "question": record.question.strip(),
                    "answer": record.answer.strip(),
                    "language": record.language
                })
                
            return training_data
            
        except Exception as e:
            print(f"Error loading training data from database: {e}")
            raise
        finally:
            db.close()
    
    def extract_links(self, text: str) -> List[str]:
        """Extract HTTP/HTTPS links from text"""
        pattern = r'https?://(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:[/?#][^\s<>"{}|\\^`\[\]]*)?'
        links = re.findall(pattern, text)
        # Clean trailing punctuation
        cleaned_links = []
        for link in links:
            link = link.rstrip('.,;:')
            cleaned_links.append(link)
        return sorted(cleaned_links)
    
    def call_api_with_retry(self, prompt: str) -> Optional[List[Dict]]:
        """Call Anthropic API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=4000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                # Response validation
                if not response or not response.content or len(response.content) == 0:
                    print("Warning: Empty response. Retrying...")
                    continue
                
                content = response.content[0].text
                if not content:
                    print("Warning: Empty response content. Retrying...")
                    continue
                
                # Clean markdown code blocks
                content = content.replace('```json', '').replace('```', '').strip()
                
                # Find and parse JSON array
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    try:
                        json_str = content[start_idx:end_idx]
                        variations = json.loads(json_str)
                        
                        # Accept 10-20 variations
                        if 10 <= len(variations) <= 20:
                            return variations
                        else:
                            print(f"Warning: Generated {len(variations)} variations, expected ~15. Retrying...")
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}. Retrying...")
                else:
                    print("Warning: JSON array not found. Retrying...")
                
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    print(f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries}). Waiting longer...")
                    time.sleep(self.retry_delay * 2)  # Wait longer for rate limits
                elif "authentication" in str(e).lower():
                    print(f"Authentication error: {e}")
                    print("Please check your API key and try again.")
                    return None  # Don't retry authentication errors
                else:
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
        
        return None
    
    def augment_qa_pair(self, qa: Dict) -> List[Dict]:
        """Augment a single question-answer pair"""
        language = qa['language'].lower()
        
        # Select appropriate prompt
        if language == 'turkish':
            prompt_template = self.turkish_prompt
        else:
            prompt_template = self.english_prompt
        
        prompt = prompt_template.format(
            question=qa['question'],
            answer=qa['answer']
        )
        
        variations = self.call_api_with_retry(prompt)
        
        if variations:
            # Format variations
            formatted_variations = []
            for i, var in enumerate(variations):
                question = var.get('question', '')
                answer = var.get('answer', '')
                
                # Link validation
                original_links = self.extract_links(qa['answer'])
                variation_links = self.extract_links(answer)
                
                if original_links != variation_links:
                    print(f"Warning: Link mismatch detected. Skipping variation.")
                    continue
                
                formatted_var = {
                    "question": question,
                    "answer": answer,
                    "language": language
                }
                formatted_variations.append(formatted_var)
            
            return formatted_variations
        
        return []
    
    def process_all_data(self):
        """Process all training data from database"""
        print("🚀 Starting CengBot Data Augmentation...")
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        last_index = checkpoint['last_processed_index']
        output_data = checkpoint['output_data']
        
        # Load training data from database
        input_data = self.load_training_data_from_db()
        print(f"📊 Loaded {len(input_data)} training records from database")
        
        # ID counter
        if output_data:
            existing_ids = [item.get('id', 0) for item in output_data if item.get('id') is not None]
            current_id = max(existing_ids) + 1 if existing_ids else 1
        else:
            current_id = 1
        
        # Process each record
        for i in range(last_index + 1, len(input_data)):
            qa = input_data[i]
            
            # Add original record
            original = {
                "id": current_id,
                "question": qa['question'],
                "answer": qa['answer'],
                "language": qa['language']
            }
            output_data.append(original)
            current_id += 1
            
            # Generate variations
            print(f"\n🔄 Processing: {i+1}/{len(input_data)} - {qa['language']} question")
            variations = self.augment_qa_pair(qa)
            
            if not variations:
                print(f"❌ ERROR: No variations generated for question {i+1}. Using original only.")
                print(f"   Question: {qa['question'][:50]}...")
                print(f"   Language: {qa['language']}")
                continue
            
            if len(variations) < 15:
                print(f"⚠️  Warning: Only {len(variations)} variations generated (expected 15)")
            
            # Add variations
            for var in variations:
                var['id'] = current_id
                output_data.append(var)
                current_id += 1
            
            # Show progress for first 10 records
            if i < 10:
                print(f"\n📝 Original {i+1}:")
                print(f"   Question: {qa['question'][:100]}...")
                print(f"   Answer: {qa['answer'][:100]}...")
                print(f"\n✨ Generated {len(variations)} variations:")
                for j, var in enumerate(variations[:3]):
                    print(f"\n   Variation {j+1}:")
                    print(f"   Question: {var['question'][:100]}...")
                    print(f"   Answer: {var['answer'][:100]}...")
                if len(variations) > 3:
                    print(f"   ... and {len(variations) - 3} more variations")
            
            # Checkpoint every 10 records
            if (i + 1) % 10 == 0:
                self.save_checkpoint(i, output_data)
                
            # Auto-save every 50 records
            if (i + 1) % 50 == 0:
                self.save_output(output_data)
                print(f"💾 Auto-save: {len(output_data)} records saved to file")
                
            # Progress report every 200 records
            if (i + 1) % 200 == 0:
                print(f"\n📊 Progress Report:")
                print(f"   Processed: {i+1}/{len(input_data)} questions")
                print(f"   Total generated: {len(output_data)} records")
                print(f"   ==========================================")
        
        # Final save
        print("\n🎉 Data augmentation completed!")
        self.save_checkpoint(len(input_data) - 1, output_data)
        self.save_output(output_data)
        
        print(f"\n📊 Final Summary:")
        print(f"   Original questions: {len(input_data)}")
        print(f"   Total generated records: {len(output_data)}")
        print(f"   Output file: {self.output_file}")
        
        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("🗑️  Checkpoint file cleaned up")
    
    def save_output(self, data: List[Dict]):
        """Save output data to JSONL file"""
        temp_file = str(self.output_file) + '.tmp'
        
        # Write to temporary file first
        with open(temp_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Atomic move to final file
        try:
            os.replace(temp_file, self.output_file)
        except AttributeError:
            # Python < 3.3 fallback
            shutil.move(temp_file, self.output_file)

def main():
    """Main function with user interaction"""
    print("🤖 CengBot Data Augmentation System")
    print("=" * 50)
    
    # Check if anthropic is available
    if not ANTHROPIC_AVAILABLE:
        print("❌ ERROR: Anthropic package not installed")
        print("Please install with: pip install anthropic")
        sys.exit(1)
    
    # Security warning
    print("\n⚠️  IMPORTANT SECURITY NOTICE:")
    print("- This script uses Anthropic's Claude API")
    print("- API usage incurs costs")
    print("- Your API key will NOT be stored anywhere")
    print("- API key is only used for this session")
    print("- Process can be interrupted safely (uses checkpoints)")
    
    # Get API key from user
    api_key = input("\n🔑 Enter your Anthropic API key: ").strip()
    
    if not api_key:
        print("❌ ERROR: API key is required")
        sys.exit(1)
    
    if not api_key.startswith('sk-ant-'):
        print("❌ ERROR: Invalid API key format")
        sys.exit(1)
    
    # Model selection
    print(f"\n🤖 Claude Model Selection:")
    print("Current model: claude-3-sonnet-20240229")
    print("For updated models, visit: https://docs.anthropic.com/claude/docs/models-overview")
    print()
    print("1. Keep current model (claude-3-sonnet-20240229)")
    print("2. Use different model")
    
    model_choice = input("\nChoose option (1-2): ").strip()
    
    if model_choice == "2":
        custom_model = input("Enter model name (e.g., claude-3-opus-20240229): ").strip()
        if custom_model:
            print(f"✅ Using custom model: {custom_model}")
        else:
            print("❌ ERROR: Model name is required")
            sys.exit(1)
    else:
        custom_model = "claude-3-sonnet-20240229"
        print(f"✅ Using default model: {custom_model}")
    
    # Confirmation
    print(f"\n📋 Process will:")
    print("1. Load approved training data from database")
    print("2. Generate 15 variations per question using Claude API")
    print("3. Save augmented data to data/cengbot_qa_augmented.jsonl")
    print("4. Use checkpoint system for safe interruption")
    
    confirm = input("\n❓ Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Process cancelled")
        sys.exit(0)
    
    # Run augmentation
    try:
        augmenter = CengBotDataAugmenter(api_key)
        # Set custom model if specified
        if custom_model != "claude-3-sonnet-20240229":
            augmenter.model_name = custom_model
        augmenter.process_all_data()
        print("\n✅ Data augmentation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        print("Checkpoint saved - you can resume later")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    
    finally:
        # Clear API key from memory
        api_key = None

if __name__ == "__main__":
    main()