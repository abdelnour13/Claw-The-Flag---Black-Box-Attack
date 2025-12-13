import dotenv
from datetime import datetime

dotenv.load_dotenv(".env")

WORKER_MEMORY = dotenv.get_key(".env", "WORKER_MEMORY")
WORKER_SWAP_MEMORY = dotenv.get_key(".env", "WORKER_SWAP_MEMORY")
WORKER_CPUS = dotenv.get_key(".env", "WORKER_CPUS")
WORKER_USE_GPU = dotenv.get_key(".env", "WORKER_USE_GPU") != "false"
WORKER_TIMEOUT = dotenv.get_key(".env", "WORKER_TIMEOUT")
MAX_REQUEST_SIZE = dotenv.get_key(".env", "MAX_REQUEST_SIZE")
MAX_UNZIPPED_SIZE = dotenv.get_key(".env", "MAX_UNZIPPED_SIZE")
END_DATE = dotenv.get_key(".env", "END_DATE")
FLAG = dotenv.get_key(".env", "FLAG")
THRESHOLD = dotenv.get_key(".env", "THRESHOLD")

assert WORKER_MEMORY is not None, "WORKER_MEMORY is not set"
assert WORKER_SWAP_MEMORY is not None, "WORKER_SWAP_MEMORY is not set"
assert WORKER_CPUS is not None, "WORKER_CPUS is not set"
assert WORKER_USE_GPU is not None, "WORKER_USE_GPU is not set"
assert WORKER_TIMEOUT is not None, "WORKER_TIMEOUT is not set"
assert MAX_REQUEST_SIZE is not None, "MAX_REQUEST_SIZE is not set"
assert END_DATE is not None, "END_DATE is not set"
assert FLAG is not None, "FLAG is not set"
assert THRESHOLD is not None, "THRESHOLD is not set"

WORKER_TIMEOUT = int(WORKER_TIMEOUT)
END_DATE = datetime.fromisoformat(END_DATE)
THRESHOLD = float(THRESHOLD)