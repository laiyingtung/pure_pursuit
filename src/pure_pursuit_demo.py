import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import matplotlib as mpl

# 指定 ffmpeg 路徑
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\FFMPEG\bin\ffmpeg.exe'

# 路徑
## path1 
# Path = [[0.0, 0.0], [0.571194595265405, -0.4277145118491421], [1.1417537280142898, -0.8531042347260006], 
#          [1.7098876452457967, -1.2696346390611464], [2.2705328851607995, -1.6588899151216996], 
#          [2.8121159420106827, -1.9791445882187304], [3.314589274316711, -2.159795566252656], 
#          [3.7538316863009027, -2.1224619985315876], [4.112485112342358, -1.8323249172947023], 
#          [4.383456805594431, -1.3292669972090994], [4.557386228943757, -0.6928302521681386], 
#          [4.617455513800438, 0.00274597627737883], [4.55408382321606, 0.6984486966257434], 
#          [4.376054025556597, 1.3330664239172116], [4.096280073621794, 1.827159263675668], 
#          [3.719737492364894, 2.097949296701878], [3.25277928312066, 2.108933125822431], 
#          [2.7154386886417314, 1.9004760368018616], [2.1347012144725985, 1.552342808106984], 
#          [1.5324590525923942, 1.134035376721349], [0.9214084611203568, 0.6867933269918683], 
#          [0.30732366808208345, 0.22955002391894264], [-0.3075127599907512, -0.2301742560363831], 
#          [-0.9218413719658775, -0.6882173194028102], [-1.5334674079795052, -1.1373288016589413], 
#          [-2.1365993767877467, -1.5584414896876835], [-2.7180981380280307, -1.9086314914221845], 
#          [-3.2552809639439704, -2.1153141204181285], [-3.721102967810494, -2.0979137913841046], 
#          [-4.096907306768644, -1.8206318841755131], [-4.377088212533404, -1.324440752295139], 
#          [-4.555249804461285, -0.6910016662308593], [-4.617336323713965, 0.003734984720118972], 
#          [-4.555948690867849, 0.7001491248072772], [-4.382109193278264, 1.3376838311365633], 
#          [-4.111620918085742, 1.8386823176628544], [-3.7524648889185794, 2.1224985058331005], 
#          [-3.3123191098095615, 2.153588702898333], [-2.80975246649598, 1.9712114570096653], 
#          [-2.268856462266256, 1.652958931009528], [-1.709001159778989, 1.2664395490411673], 
#          [-1.1413833971013372, 0.8517589252820573], [-0.5710732645795573, 0.4272721367616211], 
#          [0, 0], [0.571194595265405, -0.4277145118491421]]

# path2
# Path= [[5.0, 0.0], [4.951340343707852, 0.6958655048003272], [4.8063084796915945, 1.3781867790849958],
#          [4.567727288213004, 2.033683215379001], [4.24024048078213, 2.6495963211660243],
#          [3.83022221559489, 3.2139380484326963], [3.345653031794291, 3.715724127386971],
#          [2.7959645173537337, 4.145187862775209], [2.1918557339453875, 4.493970231495835],
#          [1.5450849718747373, 4.755282581475767], [0.8682408883346521, 4.92403876506104], 
#          [0.1744974835125054, 4.9969541350954785], [-0.5226423163382677, 4.972609476841367], 
#          [-1.2096094779983388, 4.851478631379982], [-1.8730329670795602, 4.635919272833937], 
#          [-2.499999999999999, 4.330127018922194], [-3.0783073766282913, 3.94005376803361], 
#          [-3.5966990016932563, 3.473291852294986], [-4.045084971874736, 2.9389262614623664], 
#          [-4.414737964294635, 2.3473578139294533], [-4.698463103929542, 1.7101007166283444], 
#          [-4.8907380036690284, 1.0395584540887965], [-4.987820251299121, 0.3487823687206276], 
#          [-4.987820251299121, -0.3487823687206264], [-4.890738003669028, -1.0395584540887977], 
#          [-4.698463103929543, -1.7101007166283433], [-4.414737964294634, -2.347357813929454], 
#          [-4.045084971874737, -2.938926261462365], [-3.5966990016932554, -3.4732918522949863], 
#          [-3.0783073766282905, -3.9400537680336107], [-2.500000000000002, -4.330127018922193], 
#          [-1.8730329670795616, -4.635919272833936], [-1.2096094779983388, -4.851478631379982], 
#          [-0.5226423163382667, -4.972609476841367], [0.1744974835125064, -4.9969541350954785], 
#          [0.8682408883346499, -4.924038765061041], [1.5450849718747361, -4.755282581475768], 
#          [2.191855733945387, -4.493970231495835], [2.795964517353735, -4.145187862775208], 
#          [3.345653031794289, -3.7157241273869728], [3.830222215594889, -3.213938048432698], 
#          [4.240240480782129, -2.649596321166025], [4.567727288213005, -2.0336832153790008], 
#          [4.8063084796915945, -1.3781867790849947], [4.951340343707852, -0.6958655048003294], 
#          [4.99999999, 0.005457684]]

# path3 
# Path = [[0.0, 0.0], [0.26254125768221737, 0.24209456351767736], [0.5074451373838554, 0.5079953030552422], 
#            [0.7367731535242483, 0.7949196749042016], [0.9525868205227289, 1.1000851353560608], 
#            [1.1569476527986309, 1.420709140702327], [1.351917164771287, 1.7540091472345047], 
#            [1.5395568708600316, 2.097202611244102], [1.721928285484197, 2.447506989022624], 
#            [1.9010929230631177, 2.8021397368615775], [2.079112298016126, 3.1583183110524677],
#            [2.258047924762556, 3.513260167886801], [2.4399613177217403, 3.8641827636560846], 
#            [2.626913991313014, 4.208303554651825], [2.8209674599557077, 4.542839997165526], 
#            [3.024183238069159, 4.865009547488698], [3.2386228400726953, 5.17202966191284], 
#            [3.4663477803856555, 5.461117796729466], [3.7094195734273705, 5.729491408230077], 
#            [3.969899733617173, 5.9743679527061815], [4.249749935672162, 6.193655925995525], 
#            [4.550373219440277, 6.389130409851579], [4.87297843143471, 6.56391059334107], 
#            [5.218774223302297, 6.721117014296309], [5.588969246689879, 6.863870210549613], 
#            [5.984772153244293, 6.995290719933285], [6.403911937756903, 7.114698062445519], 
#            [6.82855493580133, 7.204411821685587], [7.236518394823335, 7.242000836503269], 
#            [7.605619361805246, 7.205033726771271], [7.913674883729394, 7.071079112362298], 
#            [8.141640744330578, 6.821511826735974], [8.307537838387391, 6.482653999153017], 
#            [8.453577043466677, 6.11016193004453], [8.61827785698658, 5.7556472644240335], 
#            [8.79765540891033, 5.423646237599], [8.962602376768498, 5.090870891460973], 
#            [9.085664025221194, 4.7359868752805845], [9.157424577077258, 4.358762556761046], 
#            [9.178112779264131, 3.970248866335261], [9.148048357607152, 3.5816031653092413], 
#            [9.067551037931663, 3.203982814988996], [8.937304363973116, 2.8481501010084918], 
#            [8.769547870108253, 2.5123184744285214], [8.590186948626782, 2.179861413210749], 
#            [8.424876563386551, 1.834126837566124], [8.27198924696844, 1.480284267959676], 
#            [8.100604188581302, 1.146933234545872], [7.879375496376053, 0.8628845403986634], 
#            [7.617024357548848, 0.6233689966632185], [7.321153939445111, 0.41259849662729913], 
#            [7.000586057043968, 0.21635426845168306], [6.664047546324832, 0.022084658271586315], 
#            [6.320265243267108, -0.17554236689661832], [5.977966383503088, -0.37844408800113345], 
#            [5.645878897945066, -0.5885467869901616], [5.332730940108996, -0.807777744811905], 
#            [5.047250659512108, -1.0380642424145667], [4.798166131743291, -1.2813335607463492], 
#            [4.594205517138784, -1.539513304949591], [4.433987987395978, -1.814783968056285], 
#            [4.317929554598818, -2.1091885866515897], [4.249749935672162, -2.425684568661186], 
#            [4.236579075270964, -2.760649331083595], [4.290945618844569, -3.1002212106759786], 
#            [4.426378314849303, -3.430538544194504], [4.658404911741492, -3.73773966839534], 
#            [5.002553157977467, -4.007962920034654], [5.474350801013555, -4.227346635868616], 
#            [6.089325588306083, -4.3820291526533945], [6.863005267311377, -4.458148807145158], 
#            [7.810917585485764, -4.441843936099074], [8.948590290285567, -4.319252876270311], 
#            [10.291551129167109, -4.076513964414039], [11.855327849586719, -3.6997655372854267], 
#            [13.655448199000712, -3.175145931639642], [15.70743992486541, -2.488793484231855], 
#            [18.026830774637135, -1.6268465318172337], [20.629148495772212, -0.5754434111509465], 
#            [23.52991083572695, 0.6622803828218138], [26.74464554195769, 2.1012434982608314], 
#            [28.5464356366868, 3.0324324252]]

# path4
Path = [[0.30600467,  0.08881754], [1.18685598, -0.13555342], [2.03106468, -0.36838432], 
         [2.64426339, -0.4961379], [3.10532752, -0.80780545], [3.4460637, -0.89469696], 
         [3.70712266, -1.2375757], [4.31619604, -1.46673154], [5.03345642, -1.73659132], 
         [5.26136145, -2.01334251], [5.52772478, -2.68205328], [5.70214886, -3.4151494], 
         [5.86479363, -4.09873764], [6.00082823, -4.57367473], [6.2271974 , -5.36399653], 
         [6.46400762, -6.19077113], [6.69122547, -6.98405592], [6.89897618, -7.70937517], 
         [6.60475621, -8.49104912], [6.32245341, -9.24106195], [6.26645397, -9.38983943], 
         [6.11858061, -9.7827045 ], [5.97552643, -10.16276617], [5.91610925, -10.32062377], 
         [5.86198228, -10.46442654], [5.87838146, -10.65961999], [5.94065451, -11.40083316], 
         [6.10118844, -11.78779547], [6.25068273, -12.14814704], [6.19964703, -12.85324497], 
         [6.1464167, -13.58866334], [5.93737075, -13.84838012], [5.63797826, -14.22034262], 
         [5.08267702, -14.91024382], [4.9743857, -15.19789178], [4.91591842, -15.35319503], 
         [4.54123456, -15.56610105], [4.08139576, -15.82739452], [3.75915174, -16.01050274], 
         [3.60120701, -16.32822011], [3.39066782, -16.75173507], [3.34299694, -16.84762855], 
         [3.18637679, -17.16268143], [2.65521702, -17.68224722], [2.51132937, -18.00397872], 
         [2.61655032, -18.38938527], [2.59333947, -18.89048049], [2.55512008, -19.7155926], 
         [2.52830526, -20.29449332], [2.48832748, -21.15756708]]

# Path = [[1, 1], [1.2, 1], [1.4, 1], [1.5, 1], [1.7, 1], [2, 1]]

def stop_move(loop):
    global cycle
    if(lastFoundIndex >= len(Path) - 2): cycle += 1
    if(cycle >= loop): return 1
    return 0

# 初始化變數
currentPos = Path[0]
currentHeading = 90 #一開始朝向的方位(用度表示)330
lastFoundIndex = 0
lookAheadDis = 0.8 #0.8
linearVel = 100 #100

# 輔助函數
def pt_to_pt_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def sgn(num):
    return 1 if num >= 0 else -1

# pure pursuit
def pure_pursuit_step(path, currentPos, currentHeading, lookAheadDis, LFindex):
    currentX = currentPos[0]
    currentY = currentPos[1]
    # currentX, currentY = currentPos #車輛當前位置
    lastFoundIndex = LFindex
    startingIndex = lastFoundIndex
    goalPt = [0, 0]
    end = False
    
    for i in range(lastFoundIndex, len(path) - 1):
        if(i >= len(path) - 2): 
            end = True
        x1, y1 = path[i][0] - currentX, path[i][1] - currentY
        x2, y2 = path[i+1][0] - currentX, path[i+1][1] - currentY
        dx, dy = x2 - x1, y2 - y1
        dr = math.sqrt(dx**2 + dy**2)
        D = x1 * y2 - x2 * y1
        discriminant = (lookAheadDis**2) * (dr**2) - D**2 
        #discriminant判斷路徑是否與以車輛位置為中心lookAheaDis為半徑的圓相交
        #discriminant > 0，圓與直線有兩個交點
        #discriminant = 0，圓與直線只有一個交點
        #discriminant < 0，說明圓與直線沒有交點
        if discriminant >= 0:
            sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
            sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
            sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
            sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

            sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
            sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
            minX = min(path[i][0], path[i+1][0])
            minY = min(path[i][1], path[i+1][1])
            maxX = max(path[i][0], path[i+1][0])
            maxY = max(path[i][1], path[i+1][1])

            # if one or both of the solutions are in range
            if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

                # if both solutions are in range, check which one is better
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                    # make the decision by compare the distance between the intersections and the next point in path
                    if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2
        
                # if not both solutions are in range, take the one that's in range
                else:
                    # if solution pt1 is in range, set that as goal point
                    if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2
          
                # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
                    # update lastFoundIndex and exit
                    lastFoundIndex = i
                    break

                else:
                # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                    lastFoundIndex = i+1
        
            # if no solutions are in range
            else:
                # no new intersection found, potentially deviated from the path
                # follow path[lastFoundIndex]
                goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

    # initialize proportional controller constant
    Kp = 3
    
    #算轉彎角度
    absTargetAngle = math.atan2(goalPt[1] - currentY, goalPt[0] - currentX) * 180 / math.pi
    if absTargetAngle < 0:
        absTargetAngle += 360

    turnError = absTargetAngle - currentHeading
    if turnError > 180 or turnError < -180:
        turnError = -1 * sgn(turnError) * (360 - abs(turnError))

    turnVel = Kp * turnError

    return goalPt, lastFoundIndex, turnVel, end

#動畫
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot([p[0] for p in Path], [p[1] for p in Path], '--', color='grey')
pose, = ax.plot([], [], 'o', color='black', markersize=10)
trajectory_line, = ax.plot([], [], '-', color='orange', linewidth=4)
heading_line, = ax.plot([], [], '-', color='red')
connection_line, = ax.plot([], [], '-', color='green')

#座標系統大小
ax.axis('scaled')

xs, ys = [currentPos[0]], [currentPos[1]]


cycle = 0


def pure_pursuit_animation(frams):
    global currentPos, currentHeading, lastFoundIndex  
    
    
    if(lastFoundIndex >= len(Path) - 2):
        lastFoundIndex = 0
        
    print(lastFoundIndex, len(Path))    
    
    goalPt, lastFoundIndex, turnVel, end = pure_pursuit_step(Path, currentPos, currentHeading, lookAheadDis, lastFoundIndex)    
    
    maxLinVelfeet = 200 / 60 * np.pi * 4 / 12
    maxTurnVelDeg = 200 / 60 * np.pi * 4 / 9 * 180 / np.pi

    stepDis = linearVel / 100 * maxLinVelfeet * 50 / 1000
    currentPos[0] += stepDis * np.cos(currentHeading * np.pi / 180)
    currentPos[1] += stepDis * np.sin(currentHeading * np.pi / 180)

    heading_line.set_data([currentPos[0], currentPos[0] + 0.5 * np.cos(currentHeading / 180 * np.pi)], 
                          [currentPos[1], currentPos[1] + 0.5 * np.sin(currentHeading / 180 * np.pi)])
    connection_line.set_data([currentPos[0], goalPt[0]], [currentPos[1], goalPt[1]])

    currentHeading += turnVel / 100 * maxTurnVelDeg * 50 / 1000
    if currentHeading < 0:
        currentHeading += 360
    currentHeading = currentHeading % 360

    xs.append(currentPos[0])
    ys.append(currentPos[1])

    pose.set_data([currentPos[0]], [currentPos[1]])

    trajectory_line.set_data(xs, ys)

    if (end == True):
        print('end')
        exit()


def main():
    #pure_pursuit_animation(500)
    anim = animation.FuncAnimation(fig, pure_pursuit_animation, frames=500, interval=50)

    # 保存動畫
    #anim.save('pure_pursuit_animation3.gif', writer='pillow', fps=20)
    #anim.save('pure_pursuit_animation3.mp4', writer='ffmpeg', fps=20)

    # 顯示動畫
    #from IPython.display import HTML
    #HTML(anim.to_jshtml())
    plt.show()

# 執行主程式
if __name__ == '__main__':
    main()