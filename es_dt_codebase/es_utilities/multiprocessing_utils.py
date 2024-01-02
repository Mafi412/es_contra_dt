# Utilities for custom multiprocessing and interprocess communication

import multiprocessing as mp
import traceback
import time


class ErrorCommunicativeProcess(mp.Process):
    def __init__(self, sender_connection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sender_connection = sender_connection
        
    def run(self):
        try:
            super().run()
        except BaseException:
            self.sender_connection.send(traceback.format_exc())


def check_for_errors(receiver_connection):
    if receiver_connection.poll():
        return receiver_connection.recv()


def check_for_errors_and_wait(receiver_connection):
    traceback = check_for_errors(receiver_connection)
    if traceback:
        import sys
        print("*" * 23, "EXCEPTION FROM ONE OF THE WORKERS", "*" * 24, file=sys.stderr)
        print(traceback, file=sys.stderr)
        print("*" * 35, "EXITING", "*" * 36, file=sys.stderr)
        sys.exit(1)
    
    time.sleep(0.001)
