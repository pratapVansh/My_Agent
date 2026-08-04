from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging
import asyncio
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

class LiveKitTokenRequest(BaseModel):
    user_id: str
    room_name: str

class LiveKitTokenResponse(BaseModel):
    token: str
    url: str

active_workers = {}

@router.post("/token", response_model=LiveKitTokenResponse)
async def generate_livekit_token(request: LiveKitTokenRequest, background_tasks: BackgroundTasks):
    """
    Generate a LiveKit connection token for the given user and room.
    """
    api_key = settings.livekit_api_key
    api_secret = settings.livekit_api_secret
    ws_url = settings.livekit_url

    if not api_key or not api_secret or not ws_url:
        logger.error("LiveKit credentials not configured")
        raise HTTPException(status_code=500, detail="LiveKit not configured on server.")

    try:
        from livekit import api
        from app.livekit_worker import main as worker_main
        
        token = api.AccessToken(api_key, api_secret) \
            .with_identity(request.user_id) \
            .with_name(f"User {request.user_id}") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=request.room_name,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )).to_jwt()
            
        if request.room_name not in active_workers:
            async def run_worker():
                try:
                    await worker_main(request.room_name)
                except Exception as e:
                    logger.error(f"Worker task failed with exception: {str(e)}", exc_info=True)
                finally:
                    if request.room_name in active_workers:
                        del active_workers[request.room_name]
                        logger.info(f"Removed agent worker for room {request.room_name} from active_workers")
            
            task = asyncio.create_task(run_worker())
            active_workers[request.room_name] = task
            logger.info(f"Started agent worker for room {request.room_name}")
            
        return LiveKitTokenResponse(
            token=token,
            url=ws_url
        )
    except Exception as e:
        logger.error(f"Error generating LiveKit token: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate token")
