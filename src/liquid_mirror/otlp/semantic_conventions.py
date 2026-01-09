"""
OpenTelemetry Semantic Conventions - Standard attribute names.

Reference: https://opentelemetry.io/docs/specs/semconv/
These are the standardized attribute names that ANY OTel client will use.
"""


class ResourceAttributes:
    """Resource-level attributes (service, host, cloud, k8s)."""

    # Service
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"
    SERVICE_INSTANCE_ID = "service.instance.id"
    SERVICE_NAMESPACE = "service.namespace"

    # Deployment
    DEPLOYMENT_ENVIRONMENT = "deployment.environment"

    # Host
    HOST_NAME = "host.name"
    HOST_ID = "host.id"
    HOST_TYPE = "host.type"

    # Cloud
    CLOUD_PROVIDER = "cloud.provider"
    CLOUD_ACCOUNT_ID = "cloud.account.id"
    CLOUD_REGION = "cloud.region"
    CLOUD_AVAILABILITY_ZONE = "cloud.availability_zone"

    # Kubernetes
    K8S_CLUSTER_NAME = "k8s.cluster.name"
    K8S_NAMESPACE_NAME = "k8s.namespace.name"
    K8S_POD_NAME = "k8s.pod.name"
    K8S_DEPLOYMENT_NAME = "k8s.deployment.name"
    K8S_NODE_NAME = "k8s.node.name"


class SpanAttributes:
    """Span-level attributes."""

    # HTTP
    HTTP_METHOD = "http.method"
    HTTP_REQUEST_METHOD = "http.request.method"  # newer convention
    HTTP_URL = "http.url"
    HTTP_TARGET = "http.target"
    HTTP_ROUTE = "http.route"
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"  # newer
    HTTP_REQUEST_BODY_SIZE = "http.request.body.size"
    HTTP_RESPONSE_BODY_SIZE = "http.response.body.size"

    # Database
    DB_SYSTEM = "db.system"
    DB_NAME = "db.name"
    DB_STATEMENT = "db.statement"
    DB_OPERATION = "db.operation"

    # RPC
    RPC_SYSTEM = "rpc.system"
    RPC_SERVICE = "rpc.service"
    RPC_METHOD = "rpc.method"

    # Messaging
    MESSAGING_SYSTEM = "messaging.system"
    MESSAGING_DESTINATION = "messaging.destination"
    MESSAGING_OPERATION = "messaging.operation"

    # User/Session (custom but common)
    USER_ID = "user.id"
    USER_EMAIL = "user.email"
    SESSION_ID = "session.id"
    TENANT_ID = "tenant.id"
    DEPARTMENT = "department"


class LogAttributes:
    """Log record attributes."""

    SEVERITY_NUMBER = "severity_number"
    SEVERITY_TEXT = "severity_text"
    LOG_FILE_NAME = "log.file.name"
    LOG_FILE_PATH = "log.file.path"
    LOG_RECORD_UID = "log.record.uid"

    # Exception
    EXCEPTION_TYPE = "exception.type"
    EXCEPTION_MESSAGE = "exception.message"
    EXCEPTION_STACKTRACE = "exception.stacktrace"


class MetricNames:
    """Standard metric names."""

    # HTTP Server
    HTTP_SERVER_REQUEST_DURATION = "http.server.request.duration"
    HTTP_SERVER_ACTIVE_REQUESTS = "http.server.active_requests"

    # HTTP Client
    HTTP_CLIENT_REQUEST_DURATION = "http.client.request.duration"

    # Process
    PROCESS_CPU_TIME = "process.cpu.time"
    PROCESS_MEMORY_USAGE = "process.memory.usage"

    # Runtime (Python)
    PROCESS_RUNTIME_CPYTHON_MEMORY = "process.runtime.cpython.memory"
    PROCESS_RUNTIME_CPYTHON_GC_COUNT = "process.runtime.cpython.gc_count"
