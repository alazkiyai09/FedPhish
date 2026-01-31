"""
Email component analyzers.

Each analyzer focuses on a specific part of the email following human
cognitive processing order: sender → subject → body → URLs → attachments.
"""

from src.components.sender_analyzer import SenderAnalyzer
from src.components.subject_analyzer import SubjectAnalyzer
from src.components.body_analyzer import BodyAnalyzer
from src.components.url_analyzer import URLAnalyzer
from src.components.attachment_analyzer import AttachmentAnalyzer

__all__ = [
    "SenderAnalyzer",
    "SubjectAnalyzer",
    "BodyAnalyzer",
    "URLAnalyzer",
    "AttachmentAnalyzer",
]
