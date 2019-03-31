package filter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math4.util.Pair;

import matrix2017.TCMatrix;
import utility.TMyMatrixUtil;

public class TKalmanFilter {

	// フィルタ後の状態ベクトル
	TCMatrix fFilterX;
	// 状態ベクトルの予測値
	TCMatrix fPredictX;

	// 状態方程式
	TCMatrix fFMatrix;
	// 観測方程式
	TCMatrix fHMatrix;

	// 状態ベクトルの分散共分散行列.予測値とフィルタ後
	TCMatrix fPredictP;
	TCMatrix fFilterP;

	// システムノイズの分散共分散
	TCMatrix fQ;
	// 観測ノイズの分散共分散
	TCMatrix fR;

	// 観測残渣の分散共分散行列
	TCMatrix fS;
	TCMatrix fSInv;

	// 観測残渣
	TCMatrix fHError;

	// 計算用のテンポラリ.qとrがサイズ同じことも加味して
	// メソッドをまたいでつかうのは危険
	Map<String, TCMatrix[]> fTmp;

	// カルマンゲイン
	TCMatrix fK;

	// 状態変数を記録するか
	boolean fIsHistry = false;

	// 転置済みの行列
	TCMatrix fTFMatrix;
	TCMatrix fTHMatrix;

	// 観測ベクトルの予測値
	TCMatrix fPredictZ;


	// 乱数生成器
	Random fRng;

	// {"Q", dim(Q)}.テンポラリのkeyを生成するため
	ArrayList<Pair<String, Integer>> fKeyList;

	// 平滑化用/ログ用
	ArrayList<Map<String, TCMatrix>> fHistry;

	// 時点
	int fT;

	public TKalmanFilter() {

	}

	/**
	 * 
	 * @param initialX
	 *                初期状態ベクトル
	 * @param initialP
	 *                初期分散共分散
	 * @param Q
	 *                システムノイズの分散共分散
	 * @param R
	 *                観測ノイズの分散共分散
	 */
	public TKalmanFilter(TCMatrix initialX, TCMatrix initialP, TCMatrix Q, TCMatrix R) {
		fFilterX = initialX.clone();
		fFilterP = initialP.clone();
		fQ = Q;
		fR = R;
	}
	
	public String getDir() {
		String[] tmp = this.getClass().getName().split("\\.");
        return  "Test/" + tmp[tmp.length - 1] + "/";
	}

	/**
	 * 
	 * @param size
	 *                row * columnの形式のテンポラリ行列をsize個生成
	 * @param row
	 *                行
	 * @param column
	 *                列
	 * @return size個のテンポラリをつくって返す
	 */
	public TCMatrix[] makeTmp(int size, int row, int column) {
		TCMatrix[] dst = new TCMatrix[size];
		for (int i = 0; i < dst.length; i++) {
			dst[i] = new TCMatrix(row, column);
		}
		return dst;
	}

	/**
	 * 状態方程式を行列で与える
	 * 
	 * @param F
	 */
	public void setFMatrix(TCMatrix F) {
		fFMatrix = F;
		fTFMatrix = new TCMatrix(F.getColumnDimension(), F.getRowDimension());
		fTFMatrix.tcopyFrom(F);
	}

	/**
	 * 観測方程式を行列で与える
	 * 
	 * @param H
	 */
	public void setHMatrix(TCMatrix H) {
		fHMatrix = H;
		fTHMatrix = new TCMatrix(H.getColumnDimension(), H.getRowDimension());
		fTHMatrix.tcopyFrom(H);
	}

	/**
	 * コンストラクタから与えた情報に基づいて初期化をする. コンストラクタかreadFromをつかって必要な情報を与えておくこと
	 */
	public void initialize() {
		// seed固定
		fRng = new Random(0);

		fK = new TCMatrix(fQ.getRowDimension(), fR.getRowDimension());
		fS = new TCMatrix(fR.getRowDimension(), fR.getRowDimension());
		fSInv = fS.clone();

		fHError = new TCMatrix(fR.getRowDimension(), 1);
		// テンポラリ作成.行/列ベクトルと行列用
		fKeyList = new ArrayList<>();
		fKeyList.add(new Pair<>("Q", fQ.getRowDimension()));
		fKeyList.add(new Pair<>("R", fR.getRowDimension()));
		fKeyList.add(new Pair<>("1", 1));
		// 履歴初期化
		// Filterの初期値をコピー
		fPredictP = fFilterP.clone();
		fPredictX = fFilterX.clone();
		fPredictZ = new TCMatrix(fR.getRowDimension(), 1);
		fT = 0;
		if (fIsHistry) {
			fHistry = new ArrayList<>();
		}
	}

	/**
	 * csvに全時点のログを書き出す. ファイル名はフィールドのkeyと一致
	 * 
	 * @throws IOException
	 */
	public void outputLog() throws IOException {
		if (!fIsHistry) {
			return;
		}
		Set<String> keys = fHistry.get(0).keySet();
		for (String key : keys) {
			String dst = getDir() + key + ".csv";
			PrintWriter pw = new PrintWriter(dst);
			for (Map<String, TCMatrix> entry : fHistry) {
				TCMatrix elm = entry.get(key);
				if (elm.getRowDimension() > elm.getColumnDimension()) {
					// 縦にサイズ取らないように
					elm = elm.tclone();
				}
				String s = TMyMatrixUtil.logMatrix(elm);
				pw.println(s);
			}
			pw.close();
		}
	}

	/**
	 * 現時点のログをlistに記録する
	 */
	public void putLog() {
		Map<String, TCMatrix> entry = new HashMap<>();
		entry.put("fPredictX", fPredictX.clone());
		entry.put("fFilterX", fFilterX.clone());
		entry.put("fPredictP", fPredictP.clone());
		entry.put("fFilterP", fFilterP.clone());
		entry.put("fK", fK.clone());
		fHistry.add(entry);
		fT++;
	}

	/**
	 * テンポラリ行列をkeyから作成する.
	 */
	public void setTmp() {
		fTmp = new HashMap<>();
		for (int i = 0; i < fKeyList.size(); i++) {
			for (int j = 0; j < fKeyList.size(); j++) {
				Pair<String, Integer> p1 = fKeyList.get(i);
				Pair<String, Integer> p2 = fKeyList.get(j);
				TCMatrix[] tmp = makeTmp(2, p1.getSecond(), p2.getSecond());
				fTmp.put(p1.getFirst() + p2.getFirst(), tmp);
			}
		}
	}

	/**
	 * 観測値から状態ベクトルの予測値とフィルタ後の値を求める
	 * 
	 * @param y
	 *                観測値
	 * @throws Exception
	 */
	public void calc(TCMatrix y) throws Exception {
		if (fIsHistry && fT == 0) {
			putLog();
		}
		predict();
		correct(y);
		if (fIsHistry) {
			putLog();
		}
	}

	public void setIsHistry(boolean isHistry) {
		fIsHistry = isHistry;
	}

	public void setInitialX(TCMatrix x) {
		fFilterX = x;
	}

	/**
	 * ノイズの共分散covをつかってsrcにノイズを加える
	 * 
	 * @param src
	 * @param noise
	 * @param cov
	 */
	public void addNoise(TCMatrix src, TCMatrix cov, TCMatrix m) {
		// 他の実装みると不要っぽいがwikipediaには...
		for (int c = 0; c < src.getColumnDimension(); c++) {
			for (int r = 0; r < src.getRowDimension(); r++) {
				double base = m == null ? 0.0 : m.getValue(r);
				double weight = Math.sqrt(cov.getValue(r, r));
				double noise = weight * fRng.nextGaussian() + base;
				double val = src.getValue(r, c) + noise;
				src.setValue(r, c, val);
			}
		}
	}

	/**
	 * 予測ステップ
	 * 
	 * @throws Exception
	 *                 継承メソッドが更新に失敗した場合
	 */
	public void predict() throws Exception {
		fPredictX.times(fFMatrix, fFilterX);
		// addNoise(fX, fQ);
		fPredictP.times(fFMatrix, fFilterP).times(fTFMatrix);
		fPredictP.add(fQ);
	}

	public TCMatrix getQ() {
		return fQ;
	}

	public TCMatrix getR() {
		return fR;
	}

	/**
	 * カルマンゲインを計算する
	 */
	public void calcGain() {
		// 逆行列はnewされるので遅い
		TCMatrix[] rq = fTmp.get("RQ");
		fS.times(rq[0].times(fHMatrix, fPredictP), fTHMatrix);
		fS.add(fR);
		fSInv.copyFrom(fS);
		fSInv.inverse();
		TCMatrix[] qr = fTmp.get("QR");
		fK.times(qr[0].times(fPredictP, fTHMatrix), fSInv);
	}

	/**
	 * カルマンゲインと観測値から状態ベクトルのフィルタリングをする
	 * 
	 * @param z
	 *                観測値
	 */
	public void calcXByK(TCMatrix z) {
		// 観測残渣
		fHError.sub(z, fPredictZ);
		TCMatrix tmp[] = fTmp.get("QR");
		fFilterX.add(fPredictX, tmp[0].times(fK, fHError));
	}

	/**
	 * カルマンゲインからfPのフィルタリングをする.
	 */
	public void calcPByK() {
		TCMatrix[] tmp = fTmp.get("QQ");
		tmp[0].times(fK, fHMatrix);
		// tmp[1] <= I
		tmp[1].eye();
		fFilterP.sub(tmp[1], tmp[0]).times(fPredictP);
	}

	/**
	 * 観測ベクトルzからフィルタリングをする
	 * 
	 * @param z
	 *                観測ベクトル
	 * @throws Exception
	 *                 継承メソッドで例外
	 */
	public void correct(TCMatrix z) throws Exception {
		// 観測の予測
		fPredictZ.times(fHMatrix, fPredictX);
		calcGain();
		calcXByK(z);
		calcPByK();
	}

	public TCMatrix getGain() {
		return fK;
	}

	public TCMatrix getX() {
		return fFilterX;
	}

	public TCMatrix getP() {
		return fFilterP;
	}

	/**
	 * プロパティファイルからフィールドを読み込む.引数つきのコンストラクタ呼び出しと等価
	 * 
	 * @param br
	 * @throws IOException
	 */
	public void readFrom(BufferedReader br) throws IOException {
		fFilterX = TMyMatrixUtil.readMatrix(br);
		fFilterP = TMyMatrixUtil.readMatrix(br);
		fQ = TMyMatrixUtil.readMatrix(br);
		fR = TMyMatrixUtil.readMatrix(br);
	}
	
	public void readFunction(BufferedReader br) throws IOException {
          setFMatrix(TMyMatrixUtil.readMatrix(br));
          setHMatrix(TMyMatrixUtil.readMatrix(br));
	}

	public static void main(String[] argv) throws Exception {
		// プロパティファイルから読み込む形でも引数付きコンストラクタでもok
		TKalmanFilter kf = new TKalmanFilter();
		String property = kf.getDir() + "property.txt";
		BufferedReader br = new BufferedReader(new FileReader(property));
		kf.readFrom(br);
		kf.readFunction(br);
		br.close();
		// ログ取りたいならtrue.デフォルトではfalse
		kf.setIsHistry(true);
		// 初期化
		kf.initialize();
		// テンポラリ行列セット
		kf.setTmp();
		// 観測値をファイルから読み出し.必要なら作って与える
		BufferedReader zBr = new BufferedReader(new FileReader(kf.getDir() + "z_data.txt"));
		ArrayList<TCMatrix> zList = new ArrayList<>();
		TCMatrix z = TMyMatrixUtil.readMatrix(zBr);
		while (z != null) {
			zList.add(z);
			z = TMyMatrixUtil.readMatrix(zBr);
		}
		zBr.close();
		for (TCMatrix tmp : zList) {
			// 観測ベクトルから予測=>フィルタリング
			kf.calc(tmp);
			TCMatrix predicted = kf.fFilterX;
			TCMatrix filtered = kf.fFilterX;
			System.out.println(predicted);
			System.out.println(filtered);
		}
		// ログを書き出す
		kf.outputLog();
	}
}
