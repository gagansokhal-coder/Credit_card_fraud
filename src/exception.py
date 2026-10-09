import sys


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):

        super().__init__(error_message)

        self.error_message = self.get_detailed_error_message(
            error_message,
            error_detail
        )

    @staticmethod
    def get_detailed_error_message(
        error_message,
        error_detail: sys
    ):

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:

            file_name = exc_tb.tb_frame.f_code.co_filename

            line_number = exc_tb.tb_lineno

            detailed_message = (
                f"Error occurred in Python script: "
                f"[{file_name}] "
                f"at line number [{line_number}] "
                f"error message: [{error_message}]"
            )

        else:

            detailed_message = (
                f"Error message: [{error_message}]"
            )

        return detailed_message

    def __str__(self):

        return self.error_message