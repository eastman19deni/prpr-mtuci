import cv2


def visualize_frame_with_ids(frame, tracks, cfg):
    out = frame.copy()

    desks = cfg.get('detection', {}).get('desks', [])
    desk_counts = [0] * len(desks)

    valid_ids = set()  

    for t in tracks:
        tid = t['track_id']
        x, y, w, h = t['bbox']

        cx = x + w // 2
        cy = y + int(h * 0.75)

        if h > 260:
            continue

        inside_desk = False

        for i, desk in enumerate(desks):
            x1, y1, x2, y2 = desk['rect']
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                desk_counts[i] += 1
                inside_desk = True
                break

        if not inside_desk:
            continue

        valid_ids.add(tid)

        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f'ID:{tid}',
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    for i, desk in enumerate(desks):
        x1, y1, x2, y2 = desk['rect']
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            out,
            f"{desk.get('name', 'desk')}: {desk_counts[i]}",
            (x1, y2 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    total = len(valid_ids)

    cv2.putText(
        out,
        f'Total: {total}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    info = {
        'total': total,
        'desk_counts': desk_counts
    }

    return out, info
