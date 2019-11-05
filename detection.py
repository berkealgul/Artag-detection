import cv2


class params:
    threshConsant = 7
    threshWinSizeMax = 23
    threshWinSizeMin = 3
    threshWinSizeStep = 10
    accuracyRate = 0.02
    minAreaRate = 0.03
    maxAreaRate = 4
    minCornerDisRate = 2.5
    minMarkerDisRate = 1.5


def remove_close_candidates(candidates):
    newCandidates = list()

    for i in range(len(candidates)):
        for j in range(len(candidates)):
            # adayımızın kendisini kontrol etmesini istemeyiz
            if i == j:
                continue

            minPerimeter = min(cv2.arcLength(candidates[i], True), cv2.arcLength(candidates[j], True))

            # fc ilk köşe
            for fc in range(4):
                disSq = 0
                for c in range(4):
                    modC = (fc + c) % 4
                    dx = candidates[j][c][0][0] - candidates[i][modC][0][0]
                    dy = candidates[j][c][0][1] - candidates[i][modC][0][1]
                    disSq += dx * dx + dy * dy
                disSq /= 4

                minDisPixels = minPerimeter * params.minMarkerDisRate

                if disSq < minDisPixels * minDisPixels:
                    if cv2.contourArea(candidates[i]) > cv2.contourArea(candidates[j]):
                        newCandidates.append(candidates[i])
                    else:
                        newCandidates.append(candidates[j])

    # eğer newCandidates boş ise zaten herhangi bir filtreleme
    # olmamıştır bu yüzden eskisini döndeririz
    if len(newCandidates):
        return newCandidates
    else:
        return candidates


def has_close_corners(candidate):
    minDisSq = float("inf")

    for i in range(len(candidate)):
        dx = candidate[i][0][0] - candidate[(i+1)%4][0][0]
        dy = candidate[i][0][1] - candidate[(i+1)%4][0][1]
        dsq = dx * dx + dy * dy
        minDisSq = min(minDisSq, dsq)

    minDisPixel = candidate.size * params.minCornerDisRate
    if minDisSq < minDisPixel * minDisPixel:
        return True
    else:
        return False


def detect_candidates(grayImg):
    th = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, params.threshConsant)
    cnts = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]

    # ayıklama
    candidates = list()
    for c in cnts:
        # boyut kontrolü
        maxSize = int(max(gray.shape) * params.maxAreaRate)
        minSize = int(max(gray.shape) * params.minAreaRate)
        if c.size > maxSize or c.size < minSize:
            continue

        approxCurve = cv2.approxPolyDP(c, len(c) * params.accuracyRate, True)
        # karesellik kontrolü
        if len(approxCurve) is not 4 or cv2.isContourConvex(approxCurve) is False:
            continue

        # köşler birbirlerine çokmu yakın ona bakılır
        if has_close_corners(approxCurve):
            continue
        # testleri geçerse ekle
        candidates.append(approxCurve)

    return candidates


# ANA ALGORİTMA BAŞLANGICI
camera = cv2.VideoCapture(0)
while True:
    _, frame = camera.read()
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    candidates = detect_candidates(gray)

    if len(candidates) > 0:
        candidates = remove_close_candidates(candidates)

    cv2.drawContours(frame, candidates, -1, (0, 255, 0), 3)
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) == 27:  # esc ile çıkar
        break
