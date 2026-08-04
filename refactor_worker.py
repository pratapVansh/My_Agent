import os
import re

file_path = "app/livekit_worker.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add audio_source, audio_track, cancellation_token to ParticipantState
state_class_pattern = r'(@dataclass\nclass ParticipantState:\n[\s\S]*?audio_pipeline_task: asyncio\.Task \| None = None)\n'
new_state_class = r'''\1
    audio_source: 'rtc.AudioSource | None' = None
    audio_track: 'rtc.LocalAudioTrack | None' = None
    cancellation_token: asyncio.Event = field(default_factory=asyncio.Event)
'''
content = re.sub(state_class_pattern, new_state_class, content)

# 2. Remove global audio source, track, and lock
global_audio_pattern = r'(\s*# Setup global audio publishing for the Worker.*?\n\s*audio_source = rtc\.AudioSource\(24000, 1\)\n\s*audio_track = rtc\.LocalAudioTrack\.create_audio_track\("agent-mic", audio_source\)\n\n\s*# We use a lock.*?\n\s*# .*?\n\s*audio_lock = asyncio\.Lock\(\)\n)'
content = re.sub(global_audio_pattern, '', content, flags=re.DOTALL)

# 3. Modify run_participant_agent
# It needs to catch CancelledError and broadcast interrupted.
agent_pattern = r'(\s*async def run_participant_agent\(identity: str, transcript: str\):[\s\S]*?async with audio_lock:\n\s*await tts_streamer\.stream_to_track\(chunk_stream, audio_source\)\n\s*if room:\n\s*payload = json\.dumps\({"type": "tts_complete"}\)\.encode\("utf-8"\)\n\s*await room\.local_participant\.publish_data\(payload, topic="agent-response", reliable=True\)\n\s*logger\.info\(".*?Workflow complete for user=%s", identity\)\n\s*except asyncio\.CancelledError:\n\s*logger\.info\(".*?Agent task cancelled for %s \(Barge-in\)", identity\)\n\s*raise\n)'

def replace_agent(match):
    text = match.group(1)
    # Remove async with audio_lock
    text = re.sub(r'\s*async with audio_lock:\n\s*await tts_streamer\.stream_to_track\(chunk_stream, audio_source\)', 
                  r'\n            if state.audio_source:\n                await tts_streamer.stream_to_track(chunk_stream, state.audio_source)', text)
    
    # Replace CancelledError handling
    cancelled_block = r'''
        except asyncio.CancelledError:
            logger.info("  Agent task cancelled for %s (Barge-in)", identity)
            # Safe memory commit
            if 'accumulated' in locals() and accumulated:
                state.history.append({"role": "assistant", "content": accumulated + " [Interrupted]"})
            
            # Broadcast interrupted to UI
            if room:
                payload = json.dumps({
                    "type": "interrupted",
                    "partial_text": accumulated if 'accumulated' in locals() else ""
                }).encode("utf-8")
                # Fire and forget safely since we are inside a cancelled task context
                asyncio.create_task(room.local_participant.publish_data(payload, topic="agent-response", reliable=True))
            raise
'''
    text = re.sub(r'\s*except asyncio\.CancelledError:\n\s*logger\.info\(".*?Agent task cancelled for %s \(Barge-in\)", identity\)\n\s*raise\n', cancelled_block, text)
    
    # Also track accumulated correctly in the outer scope so we can access it in the except block
    # We will initialize accumulated = "" at the top of the try block
    text = re.sub(r'(\s*)try:', r'\1accumulated = ""\n\1try:', text)
    # In token_consumer, we can update the outer accumulated variable
    # We'll just let locals() find it if we declare nonlocal accumulated inside token_consumer
    text = re.sub(r'(\s*async def token_consumer\(\) -> AsyncIterable\[str\]:)', r'\1\n                    nonlocal accumulated', text)
    return text

content = re.sub(agent_pattern, replace_agent, content)

# 4. Modify on_participant_connected to create audio source/track and publish
connect_pattern = r'(\s*@room\.on\("participant_connected"\)\n\s*def on_participant_connected\(participant: rtc\.RemoteParticipant\) -> None:[\s\S]*?session_id=f"lk_\{room_name\}_\{participant\.identity\}"\n\s*\)\n)'

def replace_connect(match):
    return match.group(1) + r'''        
        # Create dedicated audio source/track for this participant
        audio_source = rtc.AudioSource(24000, 1)
        audio_track = rtc.LocalAudioTrack.create_audio_track(f"agent-mic-{participant.identity}", audio_source)
        participants[participant.identity].audio_source = audio_source
        participants[participant.identity].audio_track = audio_track
        
        # Publish track
        asyncio.create_task(room.local_participant.publish_track(audio_track))
'''
content = re.sub(connect_pattern, replace_connect, content)

# 5. Same for on_track_subscribed where we create state if it doesn't exist
subscribe_pattern = r'(\s*if not state:\n\s*state = ParticipantState\(\n\s*identity=participant\.identity,\n\s*session_id=f"lk_\{room_name\}_\{participant\.identity\}"\n\s*\)\n\s*participants\[participant\.identity\] = state\n)'

def replace_subscribe(match):
    return match.group(1) + r'''                
                state.audio_source = rtc.AudioSource(24000, 1)
                state.audio_track = rtc.LocalAudioTrack.create_audio_track(f"agent-mic-{participant.identity}", state.audio_source)
                asyncio.create_task(room.local_participant.publish_track(state.audio_track))
'''
content = re.sub(subscribe_pattern, replace_subscribe, content)

# 6. Add on_speech_started callback
speech_pattern = r'(\s*async def _interim_callback\(transcript: str\):\n\s*payload = json\.dumps\(\{"type": "interim_transcript", "text": transcript\}\)\.encode\("utf-8"\)\n\s*await room\.local_participant\.publish_data\(payload, topic="agent-response", reliable=False\)\n)'

def replace_speech(match):
    return match.group(1) + r'''
            async def _speech_started_callback():
                state = participants.get(participant.identity)
                if state and state.agent_task and not state.agent_task.done():
                    logger.info("  Barge-in detected (SpeechStarted): Cancelling active agent task for %s", participant.identity)
                    state.agent_task.cancel()
'''
content = re.sub(speech_pattern, replace_speech, content)

# Update DeepgramBridge instantiation
bridge_pattern = r'(bridge = DeepgramBridge\(\n\s*on_utterance_end=_bridge_callback,\n\s*on_interim_cb=_interim_callback\n\s*\))'
content = re.sub(bridge_pattern, r'bridge = DeepgramBridge(\n                on_utterance_end=_bridge_callback,\n                on_interim_cb=_interim_callback,\n                on_speech_started=_speech_started_callback\n            )', content)

# Remove the old global track publishing at the bottom
publish_pattern = r'(\s*await room\.local_participant\.publish_track\(audio_track\)\n\s*logger\.info\(".*?Worker published audio track"\)\n)'
content = re.sub(publish_pattern, r'', content)

# 7. Remove the redundant on_utterance cancel logic since speech_started handles it now
on_utterance_pattern = r'(\s*async def on_utterance\(identity: str, transcript: str\) -> None:[\s\S]*?if not state:\n\s*return\n\s*)(if state\.agent_task and not state\.agent_task\.done\(\):\n\s*logger\.info\("  Barge-in detected: Cancelling active agent task for %s", identity\)\n\s*state\.agent_task\.cancel\(\)\n\s*)(state\.agent_task = asyncio\.create_task)'

def replace_on_utterance(match):
    return match.group(1) + match.group(3)
content = re.sub(on_utterance_pattern, replace_on_utterance, content)


with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
