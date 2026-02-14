from logging.config import dictConfig
from typing import Any

LOG_CONFIG: dict[str, Any] = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            '()': 'uvicorn.logging.DefaultFormatter',
            'fmt': '%(levelname)s [%(asctime)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'default': {
            'formatter': 'default',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
    },
    'loggers': {
        'uvicorn': {'handlers': ['default'], 'level': 'INFO'},
    },
}

dictConfig(LOG_CONFIG)
