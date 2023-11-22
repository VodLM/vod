import socket


def find_available_port() -> int:
    """Find an available port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind to port 0 to let the OS assign an available port
    sock.bind(("localhost", 0))

    # Get the port that was assigned
    _, port = sock.getsockname()

    # Close the socket
    sock.close()

    return port
