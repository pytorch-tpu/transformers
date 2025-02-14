import sys
import statistics
from  . import xplane_pb2



def analyze_step_duration(file_path: str) -> float:
  xspace = xplane_pb2.XSpace()  # type: ignore

  # Read and parse the xplane proto
  with open(file_path, "rb") as f:
    print(f"Parsing {file_path}", file=sys.stderr)
    xspace.ParseFromString(f.read())

  durations = []
  event_count = 0

  for plane in xspace.planes:
    if plane.name != "/device:TPU:0":
      continue
    print(f"Plane ID: {plane.id}, Name: {plane.name}", file=sys.stderr)
    for line in plane.lines:
      if line.name != "XLA Modules":
        continue
      print(f"  Line ID: {line.id}, Name: {line.name}", file=sys.stderr)
      for event in line.events:
        name: str = plane.event_metadata[event.metadata_id].name
        secs: float = event.duration_ps / 1e12
        if name.startswith("SyncTensorsGraph."):
          durations.append(secs)
          event_count += 1
          print(
              f"    Event Metadata Name: {name}, ID: {event.metadata_id}, Duration: {secs} s",
              file=sys.stderr)

  print(f"Got {event_count} iterations", file=sys.stderr)

  if event_count == 0:
    raise ValueError("No SyncTensorsGraph events found.")

  if len(durations) < 3:
    print(
        "[Warning] Not enough SyncTensorsGraph events found to drop outliers.",
        file=sys.stderr)
    # Compute a simple average.
    return sum(durations) / len(durations), event_count

  return statistics.median(durations), event_count

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_proto_file>")
    sys.exit(1)
  proto_file_path = sys.argv[1]
  try:
    # Average SyncTensorsGraph duration.
    average_duration = analyze_step_duration(proto_file_path)
    print(f"{average_duration:.4f}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)