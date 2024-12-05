import os
import json
import time
import requests
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

# Try importing OpenAI with different methods
try:
    from openai import OpenAI
    USE_NEW_CLIENT = True
except ImportError:
    import openai
    USE_NEW_CLIENT = False

class PitchEvaluator:
    def __init__(self, openai_api_key: str, assemblyai_api_key: str):
        self.assemblyai_headers = {
            'authorization': assemblyai_api_key
        }
        
        # Initialize OpenAI client based on package version
        if USE_NEW_CLIENT:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            openai.api_key = openai_api_key
            self.openai_client = openai

    def get_evaluation_prompt(self, transcript: str) -> str:
        return f"""
        Analyze this pitch transcript and provide scores and feedback in the following JSON format. 
        For each criterion, use these scoring levels:
        - 10: Perfect/Exceptional
        - 8: Great
        - 6: Good
        - 4: Not Great
        - 2: Really Poor

        {{
            "ambitiousness": {{
                "vision": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "market_potential": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "innovation": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "scalability": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "total_ambitiousness_score": <average of above scores>
            }},
            "implementation": {{
                "technical_feasibility": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "business_model": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "go_to_market": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "team_capability": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "total_implementation_score": <average of above scores>
            }},
            "delivery": {{
                "communication_effectiveness": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "storytelling": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "authentic_expertise": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "engagement_quality": {{
                    "score": <10, 8, 6, 4, or 2>,
                    "feedback": "specific feedback explaining the score"
                }},
                "total_delivery_score": <average of above scores>
            }}
        }}
                
        Rating Criteria:

        AMBITIOUSNESS CRITERIA:
        1. Potential Reach
        - High: Global impact affecting millions
        - Medium: Regional/national impact affecting thousands
        - Low: Local impact affecting hundreds

        2. Problem Severity
        - High: Critical issues affecting health, safety, or major economic factors
        - Medium: Significant quality of life or efficiency improvements
        - Low: Non-critical improvements

        3. Problem Difficulty
        - High: Requires breakthrough innovation or solving complex challenges
        - Medium: Moderate technical challenges with some innovation needed
        - Low: Straightforward implementation using existing solutions

        IMPLEMENTATION CRITERIA:
        1. Market Competition
        - High: Emerging or underserved market with minimal competition
        - Medium: Moderate competition with room for new entrants
        - Low: Saturated market with strong established players

        2. Key Differentiator
        - High: Unique, innovative, and hard to replicate solution
        - Medium: Moderate differentiation with some unique aspects
        - Low: Limited differentiation from existing solutions

        3. Stage of Solution
        - High: Mature solution with existing clients and revenue
        - Medium: Prototype or MVP with some traction
        - Low: Idea stage with no real-world validation

        4. Financial Model
        - High: Clear, sustainable model with strong profit potential
        - Medium: Viable model needing refinement
        - Low: Unclear or potentially unsustainable model

        DELIVERY CRITERIA:
        1. Communication_effectiveness
        - High: Clear, concise, and effective communication
        - Medium: Somewhat effective but with room for improvement
        - Low: Unclear, disorganized, or difficult to follow

        2. Storytelling
        - High: Extremely clear, organized, and easy to understand
        - Medium: Moderately clear with some room for improvement
        - Low: Unclear, disorganized, or difficult to follow

        3. Authentic_expertise
        - High: Convincing, charismatic, and highly confident delivery
        - Medium: Somewhat confident but may lack conviction
        - Low: Hesitant, nervous, or lacking confidence

        4. Engagement_quality
        - High: Extremely clear, organized, and easy to understand
        - Medium: Moderately clear with some room for improvement
        - Low: Unclear, disorganized, or difficult to follow

        Scoring Guidelines:
        10 - Perfect: Exceptional, really high quality
        8 - Really Good: Strong performance with minor room for improvement
        6 - Good: Solid performance meeting expectations
        4 - Not Great: Below expectations with noticeable issues in few areas
        2 - Really Poor: Below expectations with major issues in several areas

        Analyze this pitch transcript: {transcript}
        """

    def _calculate_overall_score(self, content_analysis: Dict[str, Any]) -> float:
        try:
            # Get scores directly (they're already out of 10)
            ambitiousness_score = content_analysis['ambitiousness']['total_ambitiousness_score']
            implementation_score = content_analysis['implementation']['total_implementation_score']
            delivery_score = content_analysis['delivery']['total_delivery_score']
            
            # Add debug logging
            print(f"Delivery score calculation:")
            print(f"Raw delivery score: {delivery_score}")
            
            # Calculate weighted overall score
            overall_score = (
                (ambitiousness_score * 0.35) + 
                (implementation_score * 0.50) + 
                (delivery_score * 0.15)
            )
            
            print(f"Score Breakdown:")
            print(f"- Ambitiousness: {ambitiousness_score}")
            print(f"- Implementation: {implementation_score}")
            print(f"- Delivery: {delivery_score}")
            print(f"- Overall: {overall_score}")
            
            return round(overall_score, 2)

        except Exception as e:
            print(f"Error in calculate_overall_score: {str(e)}")
            raise Exception(f"Error calculating overall score: {str(e)}")

    def analyze_transcript(self, transcript: str, max_retries: int = 3) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                print(f"Analyzing transcript content... (Attempt {attempt + 1}/{max_retries})")
                if USE_NEW_CLIENT:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an experienced startup investor and pitch analyst, you are obective and data driven. You will provide analysis in valid JSON format only, no other text."},
                            {"role": "user", "content": self.get_evaluation_prompt(transcript)}
                        ]
                    )
                    content = response.choices[0].message.content
                else:
                    response = self.openai_client.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an experienced startup investor and pitch analyst, you are obective and data driven. You will provide analysis in valid JSON format only, no other text."},
                            {"role": "user", "content": self.get_evaluation_prompt(transcript)}
                        ]
                    )
                    content = response['choices'][0]['message']['content']
                
                print(f"Raw API response content: {content[:200]}...")  # Log first 200 chars of response
                
                if not content.strip():
                    raise ValueError("Empty response from OpenAI API")
                    
                try:
                    parsed_content = json.loads(content)
                    return parsed_content
                except json.JSONDecodeError as je:
                    print(f"JSON parsing error: {str(je)}")
                    if attempt < max_retries - 1:
                        print("Retrying after JSON parse error...")
                        time.sleep(1)  # Wait 1 second before retrying
                        continue
                    raise
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt + 2}/{max_retries})")
                    time.sleep(1)  # Wait 1 second before retrying
                    continue
                raise Exception(f"Error in transcript analysis after {max_retries} attempts: {str(e)}")

    def evaluate_pitch(self, audio_file_path: str) -> Dict[str, Any]:
        try:
            print(f"Starting to processr: {audio_file_path}")
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_file_path)
            print(f"Transcription completed: {transcript[:100]}...")
            
            # Analyze content (includes delivery analysis now)
            content_analysis = self.analyze_transcript(transcript)
            print(f"Content analysis completed: {json.dumps(content_analysis, indent=2)}")
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(content_analysis)
            print(f"Overall score calculated: {overall_score}")
            
            # Add debug logging for delivery score
            print(f"Delivery evaluation data:")
            print(json.dumps(content_analysis['delivery'], indent=2))
            
            # Compile evaluation results
            evaluation = {
                'transcript': transcript,
                'ambitiousness_evaluation': content_analysis['ambitiousness'],
                'implementation_evaluation': content_analysis['implementation'],
                'delivery_evaluation': content_analysis['delivery'],
                'overall_score': overall_score
            }
            
            return evaluation
        
        except Exception as e:
            print(f"Error in evaluate_pitch: {str(e)}")
            raise Exception(f"Error in pitch evaluation: {str(e)}")

    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file using AssemblyAI"""
        try:
            print(f"Starting transcription of {audio_file_path}...")
            
            # Upload the audio file
            upload_url = self._upload_file(audio_file_path)
            
            # Start transcription job
            transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
            json_data = {
                "audio_url": upload_url,
                "language_detection": True
            }
            
            response = requests.post(
                transcript_endpoint,
                json=json_data,
                headers=self.assemblyai_headers
            )
            response.raise_for_status()
            transcript_id = response.json()['id']
            
            # Poll for transcription completion
            polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
            while True:
                response = requests.get(polling_endpoint, headers=self.assemblyai_headers)
                response.raise_for_status()
                transcript_result = response.json()
                
                if transcript_result['status'] == 'completed':
                    return transcript_result['text']
                elif transcript_result['status'] == 'error':
                    raise Exception(f"Transcription error: {transcript_result['error']}")
                
                time.sleep(3)
                
        except Exception as e:
            raise Exception(f"Error in audio transcription: {str(e)}")

    def _upload_file(self, audio_file_path: str) -> str:
        """Upload audio file to AssemblyAI"""
        try:
            print(f"Uploading file: {audio_file_path}")
            upload_endpoint = "https://api.assemblyai.com/v2/upload"
            
            with open(audio_file_path, "rb") as audio_file:
                response = requests.post(
                    upload_endpoint,
                    headers=self.assemblyai_headers,
                    data=audio_file
                )
                response.raise_for_status()
                
            return response.json()["upload_url"]
            
        except Exception as e:
            raise Exception(f"Error uploading file: {str(e)}")


if __name__ == "__main__":
    try:
        # Replace these with your actual API keys
        OPENAI_API_KEY = "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"
        ASSEMBLYAI_API_KEY = "f56caa16052841ccbb2a1b2742c27b15"
        
        # Replace this with your audio file path
        AUDIO_FILE = "ramiro.m4a"
        print("\nInitializing Pitch Evaluator...")
        evaluator = PitchEvaluator(
            openai_api_key=OPENAI_API_KEY,
            assemblyai_api_key=ASSEMBLYAI_API_KEY
        )
        
        print("\nStarting evaluation process...")
        result = evaluator.evaluate_pitch(AUDIO_FILE)
        
        print("\nEvaluation process completed!")
        print("\nFiles generated:")
        print(f"1. Transcript: transcript_{os.path.splitext(os.path.basename(AUDIO_FILE))[0]}.txt")
        print(f"2. Full Evaluation: evaluation_{os.path.splitext(os.path.basename(AUDIO_FILE))[0]}.json")
        print(f"3. Summary: summary_{os.path.splitext(os.path.basename(AUDIO_FILE))[0]}.md")
        
        print("\nOverall Score:", result['overall_score'])
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check your API keys and file path, and try again.")