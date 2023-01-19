# System Components: Inference Stream

**Inference Stream** defines the data flow between policy workers and actor workers.

In our design, actor workers are in charge of executing `env.step()` (typically simulation), while
policy workers running `policy.rollout_step()` (typically neural network inference). The inference
stream is the abstraction of the data flow between them: the actor workers send environment
observations as requests, and the policy workers return actions as responses, both plus other
additional information.

## class `InferenceClient`

Interface used by the actor workers to obtain actions given current observation.

See [local/system/inference_stream.py](../../local/system/inference_stream.py) for detailed API description.

## class `InferenceServer`

Interface used by the policy workers to serve inference requests.

See [local/system/inference_stream.py](../../local/system/inference_stream.py) for detailed API description.

## Implementations

### Local Inference Stream (class `LocalInferenceClient`, `LocalInferenceServer`)

Local Inference Stream is a simple Inference Stream implementation with `multiprocessing.Queue`.

Multiple clients are mapped to one server. Which means in a client instance, there are a request queue and a respond queue. In a server instance, there are multiple request and respond queues.

`LocalInferenceClient` instance stores: 
- `self.__request_buffer`: requests that have not been send to server.
- `self._pending_requests`: requests that have been send to server, but not responded.
- `self._response_cache`: requests that have been replied, but not consumed by actor workers.

The Client sends responses to the Server only when `flush()` is called by actor workers. 

`LocalInferenceServer` instance stores a mapping that maps client ID to respond queues, that tells which queue server should put responds in when inference is done. 

### Inline Inference Stream (class `InlineInferenceClient`)

Inline Inference Stream is a special type of Inference Stream. It is used to do inference on CPU devices. 

GPU inference is usually faster than CPU inference, however not always more efficient. In some occasion, when GPU resource is not available or transmitting data between different processes is not optimal (due to bandwidth or efficiency problem), CPU inference is the better choice. 

This is where Inline Inference Stream comes to play. To understand Inline Inference Stream better, we can treat it as an Inference Stream that connect actor workers with "policy workers" that inference with CPU devices. However, in implementation, inference is done when calling `flush()` method in `InlineInferenceClient`.  

The implementation for inference is similar to [Policy Worker](03_policy_worker.md). The implementation for data stream is similar to normal (local) inference stream.


# Related References
- [System Overview](00_system_overview.md)
- [Actor Worker](02_actor_worker.md)
- [Policy Worker](03_policy_worker.md)

# Related Files and Directories
- [system/inference_stream.py](../../src/rlsrl/system/inference_stream.py)
- [system/basic/local_inference.py](../../src/rlsrl/system/inference_stream.py)

# What's next

- [Sample Stream](08_sample_stream.md)
