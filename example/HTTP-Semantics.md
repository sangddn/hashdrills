+++
name = "HTTP semantics"
source = "https://developer.mozilla.org/en-US/docs/Glossary/Idempotent"
+++

G: Practice the linked definition of HTTP idempotence: repeating the same
request has the same intended effect on the server as making it once.
Q: Is {{choose exactly one HTTP method from GET, HEAD, OPTIONS, PUT, DELETE,
POST, and PATCH; output only its name}} idempotent under HTTP semantics? Give
a brief reason.
A: {{for the method selected in Q, state whether HTTP defines it as idempotent
and explain the classification in one sentence; GET, HEAD, OPTIONS, PUT, and
DELETE are idempotent, while POST and PATCH are not guaranteed idempotent;
implementation bugs and response-body differences are out of scope}}
