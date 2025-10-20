from collections import deque


class RingBuffer:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("Size must be positive")
        # Use maxlen for automatic ring buffer behavior
        # Store bytes as individual integers in the deque
        self.buffer = deque(maxlen=size)
        self._capacity = size

    def append(self, item: int):
        """Append a single byte (integer 0-255) to the buffer."""
        if not isinstance(item, int) or not (0 <= item <= 255):
            raise ValueError("Item must be an integer between 0 and 255")
        self.buffer.append(item)

    def extend_bytes(self, items: bytearray):
        """
        Extend the buffer with bytes from a bytearray.
        If the buffer overflows, the oldest bytes are automatically discarded.
        """
        # The deque's maxlen will handle overflows automatically
        for byte_val in items:
            self.buffer.append(byte_val)  # Appends individual byte values (integers)

    def pop_bytes(self, size: int) -> bytearray:
        """
        Pop up to 'size' bytes from the buffer (from the left, i.e., oldest bytes first).
        Returns a bytearray containing the popped bytes.
        If the buffer contains fewer bytes than 'size', it pops all available bytes.
        """
        if size < 0:
            raise ValueError("Size to pop cannot be negative")

        # Determine how many bytes we can actually pop
        bytes_to_pop = min(size, len(self.buffer))

        if bytes_to_pop == 0:
            return bytearray()  # Return empty bytearray if buffer is empty or size is 0

        popped_bytes_list = []
        # Pop bytes one by one from the left (oldest first)
        for _ in range(bytes_to_pop):
            popped_bytes_list.append(self.buffer.popleft())

        # Convert the list of integers back to a bytearray
        return bytearray(popped_bytes_list)

    def size(self):
        """Get the current number of bytes in the buffer."""
        return len(self.buffer)

    def capacity(self):
        """Get the maximum capacity of the buffer."""
        return (
            self._capacity
        )  # Use the stored capacity, not buffer.maxlen which might be misleading after pops

    def is_empty(self):
        """Check if the buffer is empty."""
        return len(self.buffer) == 0

    def is_full(self):
        """Check if the buffer is full."""
        # Note: len(buffer) can be less than maxlen if items were popped
        # The buffer is full if its current size equals its initial capacity
        return len(self.buffer) == self._capacity

    def peek(self):
        """Peek at the next byte (oldest) without removing it."""
        if self.is_empty():
            raise IndexError("Ring buffer is empty")
        return self.buffer[0]  # Access the leftmost (oldest) byte value

    def __repr__(self):
        # Convert integers back to a readable byte representation for display
        byte_list_repr = [hex(b) for b in self.buffer]
        return f"RingBuffer(size={self.capacity()}, bytes={byte_list_repr})"


"""
python src/common/ringbuffer.py
"""
if __name__ == "__main__":
    # Example Usage
    rb = RingBuffer(10)

    # Fill the buffer beyond capacity using extend_bytes
    initial_data = bytearray(b"HelloWorld")
    rb.extend_bytes(initial_data)  # Adds 'H', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd'
    print(f"After initial extend_bytes: {rb}, Size: {rb.size()}")
    # Should show full buffer: [0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x57, 0x6f, 0x72, 0x6c, 0x64], Size: 10

    # Add more data that will cause overflow
    more_data = bytearray(b"123")
    rb.extend_bytes(more_data)  # Adds '1', '2', '3'. 'H', 'e', 'l' are overwritten.
    print(f"After adding '123': {rb}, Size: {rb.size()}")
    # Should show: [0x6f, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x31, 0x32, 0x33], Size: 10 (if capacity was 10, it stays 10, but first 3 chars were overwritten)

    # Pop 5 bytes
    popped_data = rb.pop_bytes(5)
    print(
        f"Popped bytes: {popped_data}, as string: {popped_data.decode(errors='ignore')}"
    )  # Decode for readability
    print(f"Buffer after pop_bytes(5): {rb}, Size: {rb.size()}")
    # Popped: bytearray(b'oWorl'), Size: 5

    # Pop more bytes than available
    popped_more = rb.pop_bytes(10)  # Tries to pop 10, but only 4 are available
    print(f"Popped more bytes: {popped_more}, as string: {popped_more.decode(errors='ignore')}")
    print(f"Buffer after pop_bytes(10): {rb}, Size: {rb.size()}")  # Should be empty
    print(f"Is empty: {rb.is_empty()}")  # Should print True

    # Try to pop from empty buffer
    popped_empty = rb.pop_bytes(2)
    print(f"Popped from empty buffer: {popped_empty}")  # Should print bytearray(b'')

    # Add some more bytes
    rb.extend_bytes(bytearray(b"FinalBytes"))
    print(f"After adding 'FinalBytes': {rb}, Size: {rb.size()}")

    # Pop all
    popped_all = rb.pop_bytes(rb.size())
    print(f"Popped all bytes: {popped_all}, as string: {popped_all.decode(errors='ignore')}")
    print(f"Buffer after pop all: {rb}, Size: {rb.size()}")  # Should be empty
