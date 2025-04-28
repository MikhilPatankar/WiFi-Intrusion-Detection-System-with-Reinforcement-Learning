# --- File: src/backend/app/broadcast.py ---
# NEW File: Initialize broadcaster instance

import logging
from broadcaster import Broadcast
from .core.config import settings

# Initialize broadcaster using URL from settings
# This instance will be shared across the application
# For 'memory://', it works within a single process.
# For 'redis://', it allows communication across multiple processes/workers.
broadcast = Broadcast(settings.BROADCAST_URL)
logging.info(f"Broadcaster initialized with URL: {settings.BROADCAST_URL}")
