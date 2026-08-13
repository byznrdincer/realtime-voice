import signal


def install_clean_shutdown():

    def handle_shutdown(
        signal_number,
        frame,
    ):
        raise SystemExit(0)

    signal.signal(
        signal.SIGINT,
        handle_shutdown,
    )

    signal.signal(
        signal.SIGTERM,
        handle_shutdown,
    )
