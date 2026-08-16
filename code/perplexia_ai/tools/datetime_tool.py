"""A simple datetime tool for answering current date/time questions."""

from datetime import datetime
from typing import Optional

from dateutil import parser as date_parser


class DateTimeTool:
    """A simple tool for retrieving date/time information."""

    @staticmethod
    def get_current_datetime(format: Optional[str] = None) -> str:
        """Return the current date and time as a formatted string.

        Args:
            format: Optional strftime format string. Defaults to a
                human-readable "Weekday, Month Day, Year HH:MM:SS AM/PM" format.

        Returns:
            str: The formatted current date and time.

        Examples:
            >>> DateTimeTool.get_current_datetime()
            'Wednesday, August 12, 2026 03:45:00 PM'
            >>> DateTimeTool.get_current_datetime("%Y-%m-%d")
            '2026-08-12'
        """
        now = datetime.now()
        return now.strftime(format or "%A, %B %d, %Y %I:%M:%S %p")

    @staticmethod
    def get_day_of_week(date_str: str) -> str:
        """Return the day of the week for an arbitrary date string.

        Accepts natural-language or common formatted dates (e.g. "January 17, 2026",
        "2026-01-17", "17/01/2026") and returns an error message if the date can't
        be parsed.

        Args:
            date_str: A date expressed as text, e.g. "January 17, 2026".

        Returns:
            str: The weekday name (e.g. "Saturday"), or an error message if the
                date is invalid.

        Examples:
            >>> DateTimeTool.get_day_of_week("January 17, 2026")
            'Saturday'
        """
        try:
            parsed_date = date_parser.parse(date_str, fuzzy=True)
        except (ValueError, OverflowError):
            return f"Error: Could not understand the date '{date_str}'"
        return parsed_date.strftime("%A")
