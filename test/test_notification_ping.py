import http.client
import logging
import socket
import ssl
import unittest

# Setup logging.
logging.basicConfig(level=logging.INFO)

# Credentials used for requesting to the API endpoint.
CLIENT_CERT = 'certs/cam-client1.crt'
CLIENT_KEY = 'certs/cam-client1.key'

class TestNotificationAPI(unittest.TestCase):
    API_ENDPOINT_HOST = socket.gethostbyaddr('192.168.0.5')[0]
    API_ENDPOINT_PORT = 3000

    def test_notification_api_ping(self):
        """Tests if the notification API is up and is pingable."""
        logging.info(f"Using endpoint {self.API_ENDPOINT_HOST}:{self.API_ENDPOINT_PORT}")

        request_headers = {
            'Content-Type': 'application/json',
        }

        context= ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        context.load_cert_chain(CLIENT_CERT, CLIENT_KEY)
        connection = http.client.HTTPSConnection(host=self.API_ENDPOINT_HOST, port=3000, context=context)

        connection.request(method='GET', url='/ping', headers=request_headers)
        resp = connection.getresponse()
        self.assertEqual(resp.status, 200, f'Request failed due to: {resp.reason}')

