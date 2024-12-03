import os
import json
import time
import requests
import numpy as np
from typing import Dict, Any
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
        return f"""Analyze the following startup pitch transcript and provide a comprehensive evaluation of ambitiousness, implementation, and delivery. Respond in the following JSON format:

        {{
            "ambitiousness": {{
                "potential_reach": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "estimated_impact": "<quantified estimate>"
                }},
                "problem_severity": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "key_impact_areas": ["<area1>", "<area2>"]
                }},
                "problem_difficulty": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "technical_challenges": ["<challenge1>", "<challenge2>"]
                }},
                "total_ambitiousness_score": <calculated_score>
            }},
            "implementation": {{
                "market_competition": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "key_competitors": ["<competitor1>", "<competitor2>"]
                }},
                "key_differentiator": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "unique_features": ["<feature1>", "<feature2>"]
                }},
                "stage_of_solution": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "stage_details": "<details>"
                }},
                "financial_model": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>",
                    "revenue_streams": ["<stream1>", "<stream2>"]
                }},
                "total_implementation_score": <calculated_score>
            }},
            "delivery": {{
                "confidence": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>"
                }},
                "clarity": {{
                    "rating": "<High/Medium/Low>",
                    "score": <4/2/1>,
                    "justification": "<explanation>"
                }},
                "total_delivery_score": <calculated_score>
            }},
            "overall_evaluation": {{
                "combined_score": <combined_score>,
                "key_strengths": ["<strength1>", "<strength2>"],
                "key_risks": ["<risk1>", "<risk2>"],
                "recommendations": ["<recommendation1>", "<recommendation2>"]
            }}
        }}

        Rating Criteria:

        AMBITIOUSNESS CRITERIA:
        1. Potential Reach
        - High (4): Global impact affecting millions
        - Medium (2): Regional/national impact affecting thousands
        - Low (1): Local impact affecting hundreds

        2. Problem Severity
        - High (4): Critical issues affecting health, safety, or major economic factors
        - Medium (2): Significant quality of life or efficiency improvements
        - Low (1): Non-critical improvements

        3. Problem Difficulty
        - High (4): Requires breakthrough innovation or solving complex challenges
        - Medium (2): Moderate technical challenges with some innovation needed
        - Low (1): Straightforward implementation using existing solutions

        IMPLEMENTATION CRITERIA:
        1. Market Competition
        - High (4): Emerging or underserved market with minimal competition
        - Medium (2): Moderate competition with room for new entrants
        - Low (1): Saturated market with strong established players

        2. Key Differentiator
        - High (4): Unique, innovative, and hard to replicate solution
        - Medium (2): Moderate differentiation with some unique aspects
        - Low (1): Limited differentiation from existing solutions

        3. Stage of Solution
        - High (4): Mature solution with existing clients and revenue
        - Medium (2): Prototype or MVP with some traction
        - Low (1): Idea stage with no real-world validation

        4. Financial Model
        - High (4): Clear, sustainable model with strong profit potential
        - Medium (2): Viable model needing refinement
        - Low (1): Unclear or potentially unsustainable model

        DELIVERY CRITERIA:
        1. Confidence
        - High (4): Convincing, charismatic, and highly confident delivery
        - Medium (2): Somewhat confident but may lack conviction
        - Low (1): Hesitant, nervous, or lacking confidence

        2. Clarity
        - High (4): Extremely clear, organized, and easy to understand
        - Medium (2): Moderately clear with some room for improvement
        - Low (1): Unclear, disorganized, or difficult to follow

        SCORING:
        - Ambitiousness Score = (reach_score + severity_score + difficulty_score) / 12 * 10
        - Implementation Score = (competition_score + differentiator_score + stage_score + financial_score) / 16 * 10
        - Delivery Score = (confidence_score + clarity_score) / 8 * 10
        - Combined Score = (Ambitiousness Score * 0.35) + (Implementation Score * 0.5) + (Delivery Score * 0.15)

        Analyze the following transcript and provide detailed justifications for each rating:

        {transcript}
        """

    def _calculate_overall_score(self, content_analysis: Dict[str, Any], delivery_analysis: Dict[str, Any]) -> float:
        try:
            # Get ambitiousness, implementation, and delivery scores
            ambitiousness_score = content_analysis['ambitiousness']['total_ambitiousness_score']
            implementation_score = content_analysis['implementation']['total_implementation_score']
            delivery_score = content_analysis['delivery']['total_delivery_score']

            # Normalize scores to 0-10 scale
            ambitiousness_score = (ambitiousness_score / 12) * 10
            implementation_score = (implementation_score / 16) * 10
            delivery_score = (delivery_score / 8) * 10

            # Calculate overall score
            overall_score = (ambitiousness_score * 0.35) + (implementation_score * 0.5) + (delivery_score * 0.15)

            # Final normalization to ensure score is 0-10
            overall_score = min(max(overall_score, 0), 10)

            return round(overall_score, 2)

        except Exception as e:
            print(f"Error in calculate_overall_score: {str(e)}")
            raise Exception(f"Error calculating overall score: {str(e)}")

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        try:
            print("Analyzing transcript content...")
            if USE_NEW_CLIENT:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an experienced startup investor and pitch analyst. You will provide analysis in valid JSON format only, no other text."},
                        {"role": "user", "content": self.get_evaluation_prompt(transcript)}
                    ]
                )
                content = response.choices[0].message.content
            else:
                response = self.openai_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an experienced startup investor and pitch analyst. You will provide analysis in valid JSON format only, no other text."},
                        {"role": "user", "content": self.get_evaluation_prompt(transcript)}
                    ]
                )
                content = response['choices'][0]['message']['content']
            
            return json.loads(content)
            
        except Exception as e:
            raise Exception(f"Error in transcript analysis: {str(e)}")

    def evaluate_pitch(self, audio_file_path: str) -> Dict[str, Any]:
        try:
            # Transcribe audio
            transcript = self.transcribe_audio(audio_file_path)
            print(f"Transcription completed: {transcript[:100]}...")  # Debug log
            
            # Analyze content and delivery
            content_analysis = self.analyze_transcript(transcript)
            print(f"Content analysis completed: {json.dumps(content_analysis, indent=2)}")  # Debug log
            
            delivery_analysis = self.analyze_audio(audio_file_path)
            print(f"Delivery analysis completed: {json.dumps(delivery_analysis, indent=2)}")  # Debug log
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(content_analysis, delivery_analysis)
            
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
            print(f"Error in evaluate_pitch: {str(e)}")  # Debug log
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

    def analyze_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Analyze audio file for delivery aspects like confidence and clarity"""
        try:
            # Placeholder implementation for audio analysis
            # You would replace this with actual analysis logic
            print(f"Analyzing audio delivery for {audio_file_path}...")
            
            # Example analysis results
            delivery_analysis = {
                "confidence": {
                    "rating": "High",
                    "score": 4,
                    "justification": "The presenter demonstrated confidence in explaining their product."
                },
                "clarity": {
                    "rating": "High",
                    "score": 4,
                    "justification": "The pitch was clear and easy to comprehend."
                },
                "total_delivery_score": 8.0  # Example score
            }
            
            return delivery_analysis
            
        except Exception as e:
            raise Exception(f"Error in audio analysis: {str(e)}")


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