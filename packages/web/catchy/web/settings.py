import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: list[str] | None = None) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = _path_env("CATCHY_DATA_DIR", BASE_DIR)

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "catchy-web-development-secret-key")
DEBUG = _bool_env("DJANGO_DEBUG", True)
if not DEBUG and SECRET_KEY == "catchy-web-development-secret-key":
    raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG=0")

ALLOWED_HOSTS = _list_env("DJANGO_ALLOWED_HOSTS", ["*"] if DEBUG else [])
CSRF_TRUSTED_ORIGINS = _list_env("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "catchy.web.ctf",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "catchy.web.middleware.UnhandledExceptionMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "catchy.web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "catchy" / "web" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "catchy.web.wsgi.application"
ASGI_APPLICATION = "catchy.web.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": _path_env("CATCHY_SQLITE_PATH", DATA_DIR / "db.sqlite3"),
        "OPTIONS": {
            "timeout": 30,
            "transaction_mode": "IMMEDIATE",
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=30000",
        },
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = os.environ.get("DJANGO_LANGUAGE_CODE", "ko-kr")
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "Asia/Seoul")
USE_I18N = True
USE_TZ = True

STATIC_URL = os.environ.get("DJANGO_STATIC_URL", "static/")
STATIC_ROOT = _path_env("DJANGO_STATIC_ROOT", DATA_DIR / "staticfiles")
MEDIA_URL = os.environ.get("DJANGO_MEDIA_URL", "media/")
MEDIA_ROOT = _path_env("DJANGO_MEDIA_ROOT", DATA_DIR / "media")
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGIN_REDIRECT_URL = "ctf:index"
LOGOUT_REDIRECT_URL = "login"

if _bool_env("DJANGO_TRUST_X_FORWARDED_PROTO", not DEBUG):
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = _bool_env("DJANGO_SESSION_COOKIE_SECURE", not DEBUG)
CSRF_COOKIE_SECURE = _bool_env("DJANGO_CSRF_COOKIE_SECURE", not DEBUG)
SECURE_SSL_REDIRECT = _bool_env("DJANGO_SECURE_SSL_REDIRECT", False)
SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = _bool_env(
    "DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", False
)
SECURE_HSTS_PRELOAD = _bool_env("DJANGO_SECURE_HSTS_PRELOAD", False)

LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO")
DJANGO_PLAIN_TRACEBACKS = _bool_env("DJANGO_PLAIN_TRACEBACKS", False)
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "catchy": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "django": {
            "handlers": ["console"],
            "level": os.environ.get("DJANGO_FRAMEWORK_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
        "django.security": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}

TASKS = {
    "default": {
        "BACKEND": "django.tasks.backends.immediate.ImmediateBackend",
    }
}
