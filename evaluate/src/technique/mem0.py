import os
import time
import threading
import logging
from tqdm import tqdm
from mem0 import MemoryClient
from .base import MemoryManager

logger = logging.getLogger(__name__)

class Mem0Manager(MemoryManager):  # 继承基类MemoryManager
    def __init__(self, batch_size=2):
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY")
        )
        self.batch_size = batch_size
        self.technique_name = "Mem0"

    def mem_add(self, session_data, qa_index):
        speaker_a, speaker_b = session_data['participants']
        speaker_a_user_id = f"{speaker_a}_{qa_index}"
        speaker_b_user_id = f"{speaker_b}_{qa_index}"
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)
        messages = []
        messages_reverse = []
        logger.info("处理turns")
        for turn in session_data["turns"]:
            if turn["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {turn['content']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {turn['content']}"})
            elif turn["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {turn['content']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {turn['content']}"})
            else:
                raise ValueError(f"Unknown speaker: {turn['speaker']}")
        print("处理turns完成")
        for i in tqdm(range(0, len(messages), self.batch_size), desc="Memory adding"):
            batch_messages = messages[i : i + self.batch_size]
            batch_messages_reverse = messages_reverse[i : i + self.batch_size]
            logger.info(f"Adding batch {i}")
            self.mem0_client.add(
                    batch_messages, user_id=speaker_a_user_id
                )
            self.mem0_client.add(
                    batch_messages_reverse, user_id=speaker_b_user_id
                )
