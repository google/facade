# Copyright 2025 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""String interning pool to reduce memory consumption from duplicate strings.

In Facade, principal names, resource IDs, and other identifiers appear
repeatedly across millions of events. Without interning, each occurrence
creates a new string object in memory. With string interning, we store
each unique string once and reuse references.

Example: With 10000 principals and 1M events, a principal appearing in 1000
events would normally consume 1000 * len(principal_name) bytes. With
interning, it consumes len(principal_name) bytes + 1000 pointer references.
"""

from typing import Dict


class StringPool:
    """Efficient string interning pool with separate namespaces."""
    
    def __init__(self):
        # Separate pools for different types of strings
        self._principal_pool: Dict[str, str] = {}
        self._resource_pool: Dict[str, str] = {}
        self._action_id_pool: Dict[str, str] = {}
        self._generic_pool: Dict[str, str] = {}
        
        # Also track bytes versions to avoid re-encoding
        self._bytes_pool: Dict[bytes, bytes] = {}
    
    def intern_principal(self, s: str) -> str:
        """Intern a principal name."""
        if s not in self._principal_pool:
            self._principal_pool[s] = s
        return self._principal_pool[s]
    
    def intern_resource(self, s: str) -> str:
        """Intern a resource ID."""
        if s not in self._resource_pool:
            self._resource_pool[s] = s
        return self._resource_pool[s]
    
    def intern_action_id(self, s: str) -> str:
        """Intern an action ID."""
        if s not in self._action_id_pool:
            self._action_id_pool[s] = s
        return self._action_id_pool[s]
    
    def intern(self, s: str) -> str:
        """Intern a generic string."""
        if s not in self._generic_pool:
            self._generic_pool[s] = s
        return self._generic_pool[s]
    
    def intern_bytes(self, b: bytes) -> bytes:
        """Intern bytes to avoid re-encoding the same strings."""
        if b not in self._bytes_pool:
            self._bytes_pool[b] = b
        return self._bytes_pool[b]
    
    def decode_and_intern(self, b: bytes, pool_type: str = 'generic') -> str:
        """Decode bytes and intern the result in one step."""
        s = b.decode("utf-8", "ignore")
        if pool_type == 'principal':
            return self.intern_principal(s)
        elif pool_type == 'resource':
            return self.intern_resource(s)
        elif pool_type == 'action_id':
            return self.intern_action_id(s)
        else:
            return self.intern(s)
    
    def encode_and_intern(self, s: str, pool_type: str = 'generic') -> bytes:
        """Intern string and return its encoded bytes (also interned)."""
        if pool_type == 'principal':
            interned_str = self.intern_principal(s)
        elif pool_type == 'resource':
            interned_str = self.intern_resource(s)
        elif pool_type == 'action_id':
            interned_str = self.intern_action_id(s)
        else:
            interned_str = self.intern(s)
        encoded = interned_str.encode("utf-8")
        return self.intern_bytes(encoded)
    
    def get_stats(self) -> Dict[str, int]:
        """Return statistics about the string pool."""
        return {
            'principals': len(self._principal_pool),
            'resources': len(self._resource_pool),
            'action_ids': len(self._action_id_pool),
            'generic': len(self._generic_pool),
            'bytes': len(self._bytes_pool),
            'total_strings': (len(self._principal_pool) + 
                            len(self._resource_pool) + 
                            len(self._action_id_pool) + 
                            len(self._generic_pool)),
        }
    
    def clear(self):
        """Clear all pools. Use with caution - only when processing is complete."""
        self._principal_pool.clear()
        self._resource_pool.clear()
        self._action_id_pool.clear()
        self._generic_pool.clear()
        self._bytes_pool.clear()


# Global singleton instance
_global_pool = StringPool()


def get_global_pool() -> StringPool:
    """Get the global string pool instance."""
    return _global_pool


def intern_principal(s: str) -> str:
    """Convenience function to intern using global pool."""
    return _global_pool.intern_principal(s)


def intern_resource(s: str) -> str:
    """Convenience function to intern using global pool."""
    return _global_pool.intern_resource(s)


def intern_action_id(s: str) -> str:
    """Convenience function to intern using global pool."""
    return _global_pool.intern_action_id(s)


def decode_and_intern(b: bytes, pool_type: str = 'generic') -> str:
    """Convenience function to decode and intern using global pool."""
    return _global_pool.decode_and_intern(b, pool_type)


def encode_and_intern(s: str, pool_type: str = 'generic') -> bytes:
    """Convenience function to encode and intern using global pool."""
    return _global_pool.encode_and_intern(s, pool_type)


def get_pool_stats() -> Dict[str, int]:
    """Get statistics from the global pool."""
    return _global_pool.get_stats()
