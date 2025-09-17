import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Persona(BaseModel):
    """Dataclass for a persona"""
    name: str = Field(description="Vollständiger Name")
    alias: str = Field(description="Spitzname oder Alias")
    age: int = Field(description="Alter", ge=18, le=80)
    hair_color: str = Field(description="Haarfarbe")
    eye_color: str = Field(description="Augenfarbe")
    height: str = Field(description="Körpergröße")
    occupation: str = Field(description="Beruf")
    residence: str = Field(description="Wohnort")
    personality_traits: List[str] = Field(description="Charaktereigenschaften")
    background: str = Field(description="Persönlicher Hintergrund")
    relationships: Dict[str, str] = Field(description="Wichtige Beziehungen")
    hobbies: List[str] = Field(description="Hobbys und Interessen")
    fears: List[str] = Field(description="Ängste und Sorgen")
    dreams_aspirations: List[str] = Field(description="Träume und Ziele")


class PersonGenerator:
    """Main class for generating personas and their life events"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the PersonGenerator
        
        Args:
            output_dir: Output directory for generated files
        """
        # Load environment variables
        load_dotenv()
        
        # Setup Logging
        self._setup_logging()
        
        # OpenAI Client Setup
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        
        # Rate Limiting
        self.last_api_call = 0
        self.min_delay_between_calls = 1.0  # Sekunden
        
        self.logger.info("PersonGenerator initialisiert")

    def _setup_logging(self) -> None:
        """Setup logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('generation_log.txt', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _rate_limit(self) -> None:
        """Rate limit API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_delay_between_calls:
            sleep_time = self.min_delay_between_calls - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()

    def _make_gpt_request(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Make a GPT-4 request
        
        Args:
            prompt: The prompt for GPT-4
            max_tokens: The maximum number of tokens
            
        Returns:
            The response from GPT-4
        """
        self._rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Du bist ein kreativer Schriftsteller, der realistische und detaillierte Geschichten über fiktive Personen erstellt. Achte auf Konsistenz und psychologische Glaubwürdigkeit."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.8
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"GPT-4 Request erfolgreich: {len(content)} Zeichen generiert")
            return content
            
        except Exception as e:
            self.logger.error(f"Fehler bei GPT-4 Request: {str(e)}")
            raise

    def create_persona(self) -> Persona:
        """
        Create a persona
        
        Returns:
            A Persona object with all details
        """
        self.logger.info("Beginne Persona-Erstellung...")
        
        prompt = """
        Erstelle eine vollständig ausgearbeitete fiktive Person. Die Person soll realistisch und glaubwürdig sein.
        
        Gib die Antwort im folgenden JSON-Format zurück:
        {
            "name": "Vollständiger Name",
            "alias": "Spitzname",
            "age": 35,
            "hair_color": "Haarfarbe",
            "eye_color": "Augenfarbe", 
            "height": "175cm",
            "occupation": "Beruf",
            "residence": "Stadt, Land",
            "personality_traits": ["Eigenschaft1", "Eigenschaft2", "Eigenschaft3"],
            "background": "Detaillierte Hintergrundgeschichte in 2-3 Sätzen",
            "relationships": {
                "partner": "Name und Beziehung",
                "family": "Familienstand und wichtige Familienmitglieder",
                "friends": "Wichtige Freundschaften"
            },
            "hobbies": ["Hobby1", "Hobby2", "Hobby3"],
            "fears": ["Angst1", "Angst2"],
            "dreams_aspirations": ["Traum1", "Ziel1", "Aspiration1"]
        }
        
        Achte darauf, dass alle Elemente zusammenpassen und eine kohärente Persönlichkeit ergeben.
        """
        
        response = self._make_gpt_request(prompt, max_tokens=800)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            persona_data = json.loads(json_str)
            persona = Persona(**persona_data)
            
            self.logger.info(f"Persona erstellt: {persona.name} ({persona.age} Jahre)")
            return persona
            
        except Exception as e:
            self.logger.error(f"Fehler beim Parsen der Persona: {str(e)}")
            raise

    def generate_life_circumstances(self, persona: Persona) -> List[str]:
        """
        Generate 20 detailed life circumstances for the person
        
        Args:
            persona: The persona
            
        Returns:
            A list of 20 life circumstances
        """
        self.logger.info("Generiere Lebensumstände...")
        
        circumstances = []
        
        # Split into different life categories
        categories = [
            ("Kindheit und Jugend", 4),
            ("Ausbildung und erste Berufsjahre", 4), 
            ("Beziehungen und Familie", 4),
            ("Berufliche Entwicklung", 4),
            ("Persönliche Krisen und Wendepunkte", 4)
        ]
        
        for category, count in categories:
            prompt = f"""
            Erstelle {count} detaillierte Lebensumstände aus dem Bereich "{category}" für folgende Person:
            
            Name: {persona.name}
            Alter: {persona.age}
            Beruf: {persona.occupation}
            Wohnort: {persona.residence}
            Persönlichkeit: {', '.join(persona.personality_traits)}
            Hintergrund: {persona.background}
            
            Jeder Lebensumstand soll:
            - MAXIMAL 120 Wörter lang sein (für ~1 Minute Audio)
            - Spezifische Details enthalten (Namen, Orte, Daten)
            - Emotionen und psychologische Aspekte einbeziehen
            - Zur Gesamtpersönlichkeit passen
            
            Format: Nummeriere die Umstände von 1 bis {count} und trenne sie mit "---"
            """
            
            response = self._make_gpt_request(prompt, max_tokens=1200)
            
            # Parse the response
            parts = response.split("---")
            for part in parts:
                cleaned = part.strip()
                if cleaned and len(cleaned) > 50:
                    circumstances.append(cleaned)
        
        self.logger.info(f"{len(circumstances)} Lebensumstände generiert")
        return circumstances[:20]  # Return exactly 20 life circumstances

    def generate_dreams(self, persona: Persona) -> List[str]:
        """
        Generate 40 different dreams (Albträume, Wunschträume, symbolische Träume)
        
        Args:
            persona: The persona
            
        Returns:
            A list of 40 dreams
        """
        self.logger.info("Generiere Träume...")
        
        dreams = []
        
        # Different dream categories
        dream_types = [
            ("Albträume und Angstträume", 10),
            ("Wunschträume und positive Träume", 10),
            ("Symbolische und metaphorische Träume", 10),
            ("Erinnerungsträume und Vergangenheitsbezug", 10)
        ]
        
        for dream_type, count in dream_types:
            prompt = f"""
            Erstelle {count} detaillierte Träume der Kategorie "{dream_type}" für folgende Person:
            
            Name: {persona.name}
            Persönlichkeit: {', '.join(persona.personality_traits)}
            Ängste: {', '.join(persona.fears)}
            Träume/Ziele: {', '.join(persona.dreams_aspirations)}
            Beziehungen: {persona.relationships}
            
            Jeder Traum soll:
            - MAXIMAL 100 Wörter lang sein (für ~1 Minute Audio)
            - Traumlogik und surreale Elemente enthalten
            - Die Psyche der Person widerspiegeln
            - Emotionen und Symbolik einbeziehen
            - In der Ich-Perspektive geschrieben sein
            
            Format: Nummeriere die Träume von 1 bis {count} und trenne sie mit "---"
            """
            
            response = self._make_gpt_request(prompt, max_tokens=1500)
            
            # Parse the response
            for part in parts:
                cleaned = part.strip()
                if cleaned and len(cleaned) > 40:
                    dreams.append(cleaned)
        
        self.logger.info(f"{len(dreams)} Träume generiert")
        return dreams[:40]

    def generate_daily_events(self, persona: Persona) -> List[str]:
        """
        Generate 40 daily events
        
        Args:
            persona: The persona
            
        Returns:
            A list of 40 daily events
        """
        self.logger.info("Generiere Tageserlebnisse...")
        
        events = []
        
        # Different event categories
        event_types = [
            ("Arbeitsalltag und berufliche Situationen", 10),
            ("Familien- und Beziehungsmomente", 10),
            ("Freizeit und Hobbys", 10),
            ("Zufällige Begegnungen und kleine Abenteuer", 10)
        ]
        
        for event_type, count in event_types:
            prompt = f"""
            Erstelle {count} alltägliche Erlebnisse der Kategorie "{event_type}" für folgende Person:
            
            Name: {persona.name}
            Beruf: {persona.occupation}
            Wohnort: {persona.residence}
            Hobbys: {', '.join(persona.hobbies)}
            Beziehungen: {persona.relationships}
            Persönlichkeit: {', '.join(persona.personality_traits)}
            
            Jedes Erlebnis soll:
            - MAXIMAL 110 Wörter lang sein (für ~1 Minute Audio)
            - Realistische Alltagssituationen darstellen
            - Emotionen und zwischenmenschliche Interaktionen enthalten
            - Spezifische Details und Dialoge einbeziehen
            - In der Ich-Perspektive geschrieben sein
            
            Format: Nummeriere die Erlebnisse von 1 bis {count} und trenne sie mit "---"
            """
            
            response = self._make_gpt_request(prompt, max_tokens=1500)
            
            # Parse the response
            parts = response.split("---")
            for part in parts:
                cleaned = part.strip()
                if cleaned and len(cleaned) > 50:
                    events.append(cleaned)
        
        self.logger.info(f"{len(events)} Tageserlebnisse generiert")
        return events[:40]

    def _limit_text_for_audio(self, text: str, max_words: int = 150) -> str:
        """
        Limit text for audio
        
        Args:
            text: The text to limit
            max_words: The maximum number of words
            
        Returns:
            The limited text
        """
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Limit the text to max_words
        limited_text = ' '.join(words[:max_words])
        
        # Try to cut at the sentence end
        last_period = limited_text.rfind('.')
        last_exclamation = limited_text.rfind('!')
        last_question = limited_text.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > len(limited_text) * 0.8:  # If the sentence end is in the last 20%
            return limited_text[:last_sentence_end + 1]
        else:
            return limited_text + "..."
    
    def text_to_audio(self, text: str, filename: str, voice: str = "alloy") -> bool:
        """
        Convert text to audio
        
        Args:
            text: The text to convert
            filename: The filename for the audio file
            voice: The voice to use for TTS
            
        Returns:
            True if the audio was successfully created, False otherwise
        """
        self._rate_limit()
        
        try:
            # Limit the text to max_words
            limited_text = self._limit_text_for_audio(text)
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=limited_text
            )
            
            audio_path = self.output_dir / "audio" / filename
            
            # Make sure the directory exists
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio data to file
            response.stream_to_file(audio_path)
            
            # Check if the file was created successfully
            if audio_path.exists() and audio_path.stat().st_size > 0:
                self.logger.info(f"Audio erstellt: {filename} ({audio_path.stat().st_size} Bytes)")
                return True
            else:
                self.logger.error(f"Audio-Datei {filename} wurde nicht korrekt erstellt")
                return False
            
        except Exception as e:
            self.logger.error(f"Fehler bei Audio-Erstellung für {filename}: {str(e)}")
            return False

    def save_persona_to_json(self, persona: Persona) -> None:
        """
        Save the persona to a JSON file
        
        Args:
            persona: The persona to save
        """
        persona_path = self.output_dir / "persona.json"
        
        with open(persona_path, 'w', encoding='utf-8') as f:
            json.dump(persona.model_dump(), f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Persona gespeichert: {persona_path}")

    def run_complete_generation(self) -> None:
        """
        Run the complete generation workflow
        """
        start_time = datetime.now()
        self.logger.info("=== Starte vollständige Personengenerierung ===")
        
        try:
            # 1. Create persona
            persona = self.create_persona()
            self.save_persona_to_json(persona)
            
            # 2. Generate life circumstances
            life_circumstances = self.generate_life_circumstances(persona)
            
            # 3. Generate dreams
            dreams = self.generate_dreams(persona)
            
            # 4. Generate daily events
            daily_events = self.generate_daily_events(persona)
            
            # 5. Convert text to audio
            self.logger.info("Beginne Audio-Generierung...")
            
            # Convert life circumstances to audio
            for i, circumstance in enumerate(life_circumstances, 1):
                filename = f"lebensumstaende_{i:02d}.mp3"
                self.text_to_audio(circumstance, filename)
            
            # Convert dreams to audio
            for i, dream in enumerate(dreams, 1):
                filename = f"traum_{i:02d}.mp3"
                self.text_to_audio(dream, filename)
            
            # Daily events to audio
            for i, event in enumerate(daily_events, 1):
                filename = f"tag_{i:02d}.mp3"
                self.text_to_audio(event, filename)
            
            # 6. Generate statistics
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=== Generierung abgeschlossen ===")
            self.logger.info(f"Dauer: {duration}")
            self.logger.info(f"Generierte Dateien:")
            self.logger.info(f"  - 1 Persona (JSON)")
            self.logger.info(f"  - {len(life_circumstances)} Lebensumstände (MP3)")
            self.logger.info(f"  - {len(dreams)} Träume (MP3)")
            self.logger.info(f"  - {len(daily_events)} Tageserlebnisse (MP3)")
            self.logger.info(f"  - Gesamt: {1 + len(life_circumstances) + len(dreams) + len(daily_events)} Dateien")
            
        except Exception as e:
            self.logger.error(f"Fehler während der Generierung: {str(e)}")
            raise


if __name__ == "__main__":
    generator = PersonGenerator()
    generator.run_complete_generation()
