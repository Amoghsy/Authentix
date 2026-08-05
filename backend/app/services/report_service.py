"""
report_service.py

This service handles finding, reading, and deleting generated deepfake report files
(JSON, Markdown, HTML) saved under backend/reports/.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
from backend.app.exceptions import ReportNotFoundError
from backend.app.utils.cleanup import cleanup_entire_session
from backend.app.utils.file_utils import is_safe_path

logger = logging.getLogger("backend.services.report")


class ReportService:
    """Manages looking up, reading, and clearing compiled deepfake reports."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def get_report_path(self, analysis_id: str, format_type: str) -> Path:
        """
        Locates the physical file path of a generated report for a given format.
        
        Args:
            analysis_id (str): Target analysis UUID.
            format_type (str): Requested file format: "json", "md", or "html".
            
        Returns:
            Path: The resolved absolute path of the target report file.
            
        Raises:
            ReportNotFoundError: If the analysis folder or specific report file is missing.
        """
        # Resolve target folder path
        session_reports_dir = self.settings.REPORTS_DIR / analysis_id
        
        # Verify folder path safety
        if not is_safe_path(self.settings.REPORTS_DIR, session_reports_dir):
            raise ReportNotFoundError("Invalid report lookup reference.")
            
        if not session_reports_dir.exists() or not session_reports_dir.is_dir():
            raise ReportNotFoundError(f"Report directory for analysis ID {analysis_id} not found.")

        # Resolve filename based on format mapping
        fmt = format_type.lower().strip()
        if fmt == "json":
            filename = "fusion_result.json"
        elif fmt == "md" or fmt == "markdown":
            filename = "fusion_report.md"
        elif fmt == "html":
            filename = "fusion_report.html"
        else:
            raise ReportNotFoundError(f"Unsupported report format requested: {format_type}")

        report_file_path = session_reports_dir / filename
        
        # Verify file path safety and existence
        if not is_safe_path(session_reports_dir, report_file_path):
             raise ReportNotFoundError("Invalid report lookup reference.")

        if not report_file_path.exists() or not report_file_path.is_file():
            raise ReportNotFoundError(f"Report file {filename} for analysis {analysis_id} not found.")

        return report_file_path.resolve()

    def read_json_report(self, analysis_id: str) -> str:
        """
        Reads the saved fusion_result.json raw string content.
        """
        path = self.get_report_path(analysis_id, "json")
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read JSON report file at {path}: {e}")
            raise ReportNotFoundError("Failed to retrieve the report contents.")

    def delete_analysis_report(self, analysis_id: str) -> None:
        """
        Deletes all reports and temporary workspace assets associated with an analysis session.
        """
        logger.info(f"Triggering purge for all session data: {analysis_id}")
        cleanup_entire_session(analysis_id, self.settings)


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/report_service.py...")
    import tempfile
    import shutil
    from typing import Optional

    try:
        temp_dir = Path(tempfile.mkdtemp())
        cfg = Settings(REPORTS_DIR=temp_dir)
        service = ReportService(cfg)
        
        analysis_id = "test-uuid-999"
        session_dir = temp_dir / analysis_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy report files
        (session_dir / "fusion_result.json").write_text('{"status": "ok"}', encoding="utf-8")
        (session_dir / "fusion_report.md").write_text("# Report Markdown", encoding="utf-8")
        
        # Test lookups
        p_json = service.get_report_path(analysis_id, "json")
        print(f"Json report path: {p_json}")
        assert p_json.name == "fusion_result.json"
        
        p_md = service.get_report_path(analysis_id, "md")
        print(f"MD report path: {p_md}")
        assert p_md.name == "fusion_report.md"
        
        content = service.read_json_report(analysis_id)
        assert "ok" in content
        print("Report lookup and content check: PASSED")
        
        # Test full deletion
        service.delete_analysis_report(analysis_id)
        assert not session_dir.exists()
        print("Report deletion: PASSED")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("ReportService self-test: PASSED")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
