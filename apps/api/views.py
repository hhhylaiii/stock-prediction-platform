from rest_framework.views import APIView


class ReadOnlyView(APIView):
    """GET and HEAD only; every other method answers 405 with Allow: GET, HEAD."""

    http_method_names = ("get", "head")
