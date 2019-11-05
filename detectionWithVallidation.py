import cv2
import numpy as np

# satır 964
# satır 601  510  eb411

'''
Yapılacaklar 

köşe dizmeyi düzelt
adayları onayla
çoklu ve tekli tespit etme modlarını uyarla 
sonuçları ros üzerinden yayınla

'''


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
    bitCellFillRate = 0.55
    monoDetection = True


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


# HATALI ÇALIŞIYOR (gereksiz de olabilir)
def sort_corners(corners):
   dx1 = corners[1][0] - corners[0][0]
   dy1 = corners[1][1] - corners[0][1]
   dx2 = corners[2][0] - corners[0][0]
   dy2 = corners[2][1] - corners[0][1]

   crossproduct = (dx1 * dy2) - (dy1 * dx2)

   if crossproduct > 0:
       corners[1], corners[3] = corners[3], corners[1]

    # deneme amaçlıdırlar (m y k
   global frame
   cv2.circle(frame, tuple(corners[0]), 2, (255, 0, 0), 3)    # mavi
   cv2.circle(frame, tuple(corners[1]), 2, (255, 255, 0), 3)  # sarı
   cv2.circle(frame, tuple(corners[2]), 2, (255, 0, 255), 3)  # mor
   cv2.circle(frame, tuple(corners[3]), 2, (0, 255, 255), 5)  # turkuaz


def get_corners(candidate):
    corners = np.array([
        [candidate[0][0][0], candidate[0][0][1]],
        [candidate[1][0][0], candidate[1][0][1]],
        [candidate[2][0][0], candidate[2][0][1]],
        [candidate[3][0][0], candidate[3][0][1]],
    ], dtype="float32")
    return corners


def get_candate_img(candidate, frame):
    corners = get_corners(candidate)
    #sort_corners(corners)

    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))#

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth-1, maxHeight-1],
         [0, maxHeight - 1]
         ], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), borderMode=cv2.INTER_NEAREST)

    return warped


def validate_candidate(candidate, frame):
    candidate_img = get_candate_img(candidate, frame)
    ret, candidate_img = cv2.threshold(candidate_img, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bits = extract_bits(candidate_img)
    recreate_img(bits)

    # gösterim amaçlı
    cv2.imshow("otsu", candidate_img)


# extract_bits fonksiyonu çalışmasını kontrol etmek için oluşturuldu
def recreate_img(bits):
    cellSize = 30
    img = np.zeros((bits.shape[0] * cellSize, bits.shape[1] * cellSize, 1))

    for j in range(bits.shape[0]):
        for i in range(bits.shape[1]):
            if bits[j, i] == 0:
                continue
            for x in range(cellSize):
                ix = i * cellSize + x
                for y in range(cellSize):
                    iy = j * cellSize + y
                    img[iy, ix] = 255

    cv2.imshow("newTag", img)


def extract_bits(img):
    # artag boyutları şimdilik fonksiyon içinde tanımlı
    # ileride belli bir parametreye bağlanacak
    markerSize = 7
    borderSize = 2

    markerSizeWithBorders = markerSize + 2 * borderSize

    bitmap = np.zeros((markerSizeWithBorders, markerSizeWithBorders))

    cellWidth = int(img.shape[1] / markerSizeWithBorders)
    cellHeight = int(img.shape[0] / markerSizeWithBorders)
    totalPixelsEachBit = cellHeight * cellWidth

    # her bit için
    for i in range(markerSizeWithBorders):
        for j in range(markerSizeWithBorders):
            # doluluk oranını hesapla
            filledPixel = 0
            for x in range(cellWidth):
                ix = cellWidth * i + x
                for y in range(cellHeight):
                    iy = cellHeight * j + y
                    if img[iy][ix] == 255:
                        filledPixel += 1

            if filledPixel / totalPixelsEachBit > params.bitCellFillRate:
                bitmap[j][i] = 1

    return bitmap


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


def find_center(marker):
    (x, y), r = cv2.minEnclosingCircle(marker)
    return int(x), int(y)


# ANA ALGORİTMA BAŞLANGICI
camera = cv2.VideoCapture(0)
while True:
    _, frame = camera.read()
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    candidates = detect_candidates(gray)

    if len(candidates) > 0:
        candidates = remove_close_candidates(candidates)
        validate_candidate(candidates[0], gray)

        if params.monoDetection is True and len(candidates) > 1:
            biggest = max(candidates, key=cv2.contourArea)
            candidates.clear()
            candidates.append(biggest)
            center = find_center(biggest)
            cv2.circle(frame, center, 5, (0,0,255), -1)

        cv2.drawContours(frame, candidates, -1, (0, 255, 0), 1)

        center = find_center(candidates[0])
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        print(len(candidates))

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:  # esc ile çıkar
        break