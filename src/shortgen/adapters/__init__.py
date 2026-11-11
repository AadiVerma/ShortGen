"""Concrete model adapters, one module per model.

Nothing here is imported by the pipeline directly: the registry imports an
adapter by name only when the configuration selects it, and heavy libraries
are imported inside ``load()`` so that importing shortgen stays instant.
"""
