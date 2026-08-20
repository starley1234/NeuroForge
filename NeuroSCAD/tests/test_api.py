import unittest

try:
    from fastapi.testclient import TestClient
    from neuroscad.api import app
except (ImportError, RuntimeError):
    TestClient = None

@unittest.skipIf(TestClient is None, "API extras are not installed")
class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.client = TestClient(app)

    def test_generate_compile_and_static_ui(self):
        response = self.client.post("/v1/generate", json={"prompt": "Кронштейн на трубу 25 мм, винт М4"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-request-id", response.headers)
        payload = response.json()
        compiled = self.client.post("/v1/compile", json={"ir": payload["ir"], "overrides": {"tube_d": 30}})
        self.assertEqual(compiled.status_code, 200)
        self.assertIn("tube_d = 30", compiled.json()["openscad"])
        self.assertIn("NEUROSCAD", self.client.get("/").text)

    def test_rejects_out_of_range_override_and_extra_fields(self):
        generated = self.client.post("/v1/generate", json={"prompt": "pipe clamp 25 mm M4"}).json()
        response = self.client.post("/v1/compile", json={"ir": generated["ir"], "overrides": {"tube_d": 999}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/v1/generate", json={"prompt": "pipe clamp", "unknown": 1}).status_code, 422)
