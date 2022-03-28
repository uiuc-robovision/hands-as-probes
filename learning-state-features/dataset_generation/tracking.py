import numpy as np
from scipy.optimize import linear_sum_assignment


CONF_THRESH = 0.2
FRAME_BUFFER = 8
IOU_THRESH = 0.4
MIN_TRACK_LENGTH = 32

class Track:
    def __init__(self, bbox, frame_id, box_id):
        self.bbox = bbox
        self.len = 1
        self.buffer = 0
        self.track = [[frame_id, box_id]]

    def update(self, bbox, frame_id=None, box_id=None):
        if bbox is not None:
            self.bbox = bbox
            self.len += 1
            self.track.append([frame_id, box_id])
        else:
            self.buffer += 1

    def to_dict(self, id):
        dct = vars(self)
        dct["id"] = id
        return dct


def bb_intersection_over_union(boxA, boxB):
    # Taken from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = [boxA.left, boxA.top, boxA.right, boxA.bottom]
    boxB = [boxB.left, boxB.top, boxB.right, boxB.bottom]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max((xB - xA, 0)) * max((yB - yA), 0)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_cost_matrix(bblist1, bblist2):
    cost = np.zeros((len(bblist1), len(bblist2)))
    for i, b1 in enumerate(bblist1):
        for j, b2 in enumerate(bblist2):
            if b2.score > CONF_THRESH:
                cost[i, j] = bb_intersection_over_union(b1.bbox, b2.bbox)
            else:
                cost[i, j] = 0
    return -cost


def get_tracks(video_id, detections, start_frame=0, end_frame=None):
    global CONF_THRESH, FRAME_BUFFER, MIN_TRACK_LENGTH, IOU_THRESH
    if end_frame is None:
        end_frame = len(detections) - 1
    next_objects = [(i, dobj) for i, dobj in enumerate(detections[start_frame].objects) if dobj.score > CONF_THRESH]
    active_tracks = [Track(dobj, start_frame, i) for i, dobj in next_objects]
    complete_tracks = []

    for i, det in enumerate(detections[start_frame: end_frame], start_frame):
        next_objects = detections[i+1].objects
        # next_hands = detections[i+1].hands
        confident_bboxes = np.array([k for k, do in enumerate(next_objects) if do.score > CONF_THRESH])

        # get cost matrix, unconfident boxes have IoU set to 0
        # Later filtered if assigned
        C = get_cost_matrix([t.bbox for t in active_tracks], next_objects)
        track_id, bbox_id = linear_sum_assignment(C)
        match_mask = (-C[track_id, bbox_id] > IOU_THRESH)

        # find ids for matched tracks, unmatched tracks and unmatched bbox
        matched_track_id, matched_bbox_id = track_id[match_mask], bbox_id[match_mask]
        unmatched_track_id = np.arange(len(active_tracks))[
            np.logical_not(np.isin(np.arange(len(active_tracks)), matched_track_id))]
        unmatched_bbox_id = confident_bboxes[
            np.logical_not(np.isin(confident_bboxes, matched_bbox_id))]

        assert matched_track_id.shape[0] + unmatched_track_id.shape[0] == len(active_tracks)
        assert np.isin(matched_bbox_id, confident_bboxes).sum() == len(matched_bbox_id)

        # update matched active_tracks
        for b, t in zip(matched_bbox_id, matched_track_id):
            active_tracks[t].update(next_objects[b], i+1, b)

        # update unmatched tracks, remove if stale
        pop_tracks = []
        for track_id in unmatched_track_id:
            active_tracks[track_id].update(None)
            if active_tracks[track_id].buffer > FRAME_BUFFER:
                if active_tracks[track_id].len > MIN_TRACK_LENGTH:
                    complete_tracks.append(active_tracks[track_id])
                pop_tracks.append(track_id)
        active_tracks = [track for t, track in enumerate(active_tracks) if t not in pop_tracks]

        # add new tracks if some objects remain unmatched
        for b in unmatched_bbox_id:
            active_tracks.append(Track(next_objects[b], i+1, b))
            assert len(detections[i+1].objects) > b

    # Close all active tracks and add mark long tracks as complete
    for track in active_tracks:
        if track.len > MIN_TRACK_LENGTH:
            complete_tracks.append(track)

    return [ct.to_dict(i) for i, ct in enumerate(complete_tracks)]


if __name__ == "__main__":
    get_tracks("P01_103")
