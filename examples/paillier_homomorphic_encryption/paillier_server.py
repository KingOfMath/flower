from flwr.server import Server

import phe as paillier


class PaillerServer(Server):

    def generate_key(self, key_length):
        keypair = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.priv_key = keypair
