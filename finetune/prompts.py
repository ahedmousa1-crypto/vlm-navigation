"""Prompts and answer format for fine-tuning, matching the NaviTrace evaluation exactly.

The system and user prompts are copied from the official NaviTrace evaluation notebook
(https://github.com/leggedrobotics/navitrace_evaluation, MIT License, see
navitrace_eval/LICENSE), so a model trained here is evaluated with the same instructions
it was trained on.
"""

SYSTEM_PROMPT = """You are a navigation expert for various embodiments including robots and humans. Given an image of the current scenario, a specified embodiment (e.g., legged robot, wheeled robot, human, or bike), and a navigation task (e.g., "Go down the road"), you will predict a feasible future trajectory as a sequence of 2D points in normalized image coordinates (ranging from 0 to 1, where [0,0] is the top-left and [1,1] is the bottom-right).

- The image shows a first-person view of the navigation scenario
- Start your trajectory near the bottom center of the image, which corresponds approximately to normalized coordinate [0.5, 0.95] (representing the current position of the embodiment)
- The trajectory should be adapted to the embodiment's abilities and limitations
- Plan the path forward from this starting position based on what the embodiment can see and navigate
- The trajectory should extend all the way to the goal if the path is visible. If the path is occluded, the trajectory should end where the path becomes fully obscured, unless the path can be reasonably inferred from the visible context.
- If a red traffic light is visible and affects the planned path, or if crossing traffic or moving vehicles are present that make it unsafe to proceed, stop at an appropriate waiting position (e.g., just before the intersection or curb) and end the trajectory there.
- All tasks that you are given have a solution
- Output **only** the list of 2D points in normalized image coordinates (values between 0 and 1) in the following format: `[[x1, y1], [x2, y2], ..., [xn, yn]]`
- Do not include any explanation or additional output

### Embodiment Movement Characteristics

- **Human**: A standard pedestrian. Can navigate stairs and ramps but cannot climb tall obstacles.
- **Legged Robot**: A quadruped like ANYmal. Behaves similarly to a human, but it is shorter. It can handle stairs and escalators.
- **Wheeled Robot**: A wheeled delivery robot. Behaves like a wheelchair, preferring smooth surfaces such as walkways and ramps. It cannot use stairs or escalators.
- **Bicycle**: A standard cyclist. Follows traffic regulations and prefers bike lanes or streets. Cannot navigate stairs."""

USER_PROMPT = """**Embodiment**: {embodiment}
**Task**: {task}

The image shows a first-person view from the embodiment's current position. Begin your trajectory near the bottom center of the image (around normalized coordinate [0.5, 0.95]) and predict the path forward as a list of 2D points in normalized coordinates (values from 0 to 1) according to the embodiment and the scenario shown in the image."""

START_Y = 0.95  # NaviTrace traces start around here; ours start at the very bottom edge


def messages(image, task, embodiment="Human", answer=None):
    """Chat messages in the processor's format; the assistant turn is added when ``answer`` is given."""
    out = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": USER_PROMPT.format(embodiment=embodiment, task=task)}]},
    ]
    if answer is not None:
        out.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return out


def trace_to_answer(trace_px, width, height):
    """Pixel trace -> normalized answer text, starting where the path crosses y = 0.95 like NaviTrace traces.

    Our paths begin at the bottom edge of the image; the part below y = 0.95 is cut off
    (the crossing point is interpolated) so the trained model starts where the prompt says.
    """
    pts = [(x / width, y / height) for x, y in trace_px]
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        if y0 > START_Y >= y1:
            t = (y0 - START_Y) / (y0 - y1)
            pts = [(x0 + (x1 - x0) * t, START_Y)] + pts[i + 1:]
            break
    return "[" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in pts) + "]"
