package filter;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Map;

import org.matheclipse.core.eval.ExprEvaluator;

import matrix2017.TCMatrix;
import utility.TMyMatrixUtil;

public abstract class TNonLinearKF extends TKalmanFilter {
	// 文字列形式の非線形関数を評価するcas
	ExprEvaluator fUtil;

	// 変数と状態ベクトルの対応.varList[i] = vector[i]の示す変数名
	String[] fVarList;

	// 状態方程式の文字列形式.次元分用意
	String[] fF;
	// 観測方程式の文字列形式.次元分用意
	String[] fH;

	// 標本点=シグマポイント/アンサンブルの集合の行列
	// (dim,点の総数)の形式
	TCMatrix fPredictXi;
	TCMatrix fFilterXi;

	// 転置
	TCMatrix fTPredictXi;

	// fPredictXiの観測変換
	TCMatrix fPredictGamma;

	// 標本点の数
	int fXiSize;

	// ベクトル向けの標本点の重み
	TCMatrix fXW;
	// 分散共分散向けの標本点の重み
	TCMatrix fPW;

	// gammaの分散
	TCMatrix fZP;

	// シグマポイントの観測ノイズの共分散
	TCMatrix fRHat;

	public TNonLinearKF() {
	}

	@Override
	public void readFrom(BufferedReader br) throws IOException {
		super.readFrom(br);
		// 関数を読み取る
		readFunction(true, br);
		readFunction(false, br);
	}

	// 文字列/行列から読み取れる.
	public void readFunction(boolean isF, BufferedReader br) throws IOException {
		// True \n var,... \n func,...
		String isString = br.readLine();
		if (isString.equals("True")) {
			String vars = br.readLine();
			String functions = br.readLine();
			vars = vars.replaceAll(" ", "");
			functions = functions.replaceAll(" ", "");
			String[] splits = functions.split(",");
			fVarList = vars.split(",");
			if (isF) {
				fF = splits;
			}
			else {
				fH = splits;
			}
		}
		else {
			TCMatrix tmp = TMyMatrixUtil.readMatrix(br);
			if (isF) {
				setFMatrix(tmp);
			}
			else {
				setHMatrix(tmp);
			}
		}
	}

	public TNonLinearKF(TCMatrix initialX, TCMatrix initialP, TCMatrix Q, TCMatrix R) {
		super(initialX, initialP, Q, R);
	}

	/**
	 * 文字列形式の状態方程式をセット
	 * 
	 * @param f
	 */
	public void setF(String[] f) {
		fF = f;
	}

	/**
	 * 方程式の評価に必要なベクトルと変数の対応関係をセット
	 * 
	 * @param varList
	 */
	public void setVarList(String[] varList) {
		fVarList = varList;
	}

	/**
	 * 標本点の観測変換をする
	 */
	public void calcGamma() {
		for (int i = 0; i < fXiSize; i++) {
			// 対象列を取り出す
			TCMatrix[] src = fTmp.get("Q1");
			TCMatrix[] dst = fTmp.get("R1");
			src[0].copySubmatrixFrom(fPredictXi, 0, fPredictXi.getRowDimension() - 1, i, i, 0, 0);
			applyFunction(src[0], dst[0], fHMatrix, fH);
			// addNoise(fGammaCalc2, fRNoise, fR);
			// 更新後のベクトルを反映
			fPredictGamma.copyAtColumn(dst[0], i);
		}
	}

	@Override
	public void initialize() {
		super.initialize();
		fRHat = new TCMatrix(fR);
		fUtil = new ExprEvaluator(true, 0);
		fPredictXi = new TCMatrix(fQ.getRowDimension(), fXiSize);
		fFilterXi = fPredictXi.clone();
		fPredictGamma = new TCMatrix(fR.getRowDimension(), fXiSize);
		fZP = new TCMatrix(fR.getRowDimension(), fR.getRowDimension());
		calcWeight();
	}

	@Override
	public void calcGain() {
		calcXByXi(fPredictGamma, fPredictZ);
		calcCovByXi(false, fPredictGamma, fPredictZ, false, fPredictGamma, fPredictZ, fZP);
		//
		// fZP.add(fR);pyFilter.平滑化時に不敵説
		// 次のメソッドでQRは使用できない
		TCMatrix pXZ = fTmp.get("QR")[0];
		calcCovByXi(true, fPredictXi, fPredictX, false, fPredictGamma, fPredictZ, pXZ);
		TCMatrix tmp = fTmp.get("RR")[0];
		// wikipedia.fZP自体にRは加えない
		tmp.copyFrom(fZP).add(fRHat).inverse();
		fK.times(pXZ, tmp);
	}

	@Override
	public void correct(TCMatrix z) throws Exception {
		calcGamma();// 標本を観測次元へ
		calcGain();
		calcXByK(z);
		calcPByK();
	}

	@Override
	/**
	 * 予測ステップ.標本点をノイズ無考慮で更新するだけ
	 * 
	 * @throws Exception
	 *                 更新関数の評価に失敗した
	 */
	public void predict() throws Exception {
		// 標本点の予測値を計算するだけ
		// 標本点にノイズを加えるかはサブクラスで
		for (int i = 0; i < fXiSize; i++) {
			TCMatrix src = fTmp.get("Q1")[0];
			src.copySubmatrixFrom(fFilterXi, 0, src.getRowDimension() - 1, i, i, 0, 0);
			TCMatrix dst = fTmp.get("Q1")[1];
			applyFunction(src, dst, fFMatrix, fF);
			fPredictXi.copyAtColumn(dst, i);
		}
	}

	/**
	 * 標本点の重み和を求める
	 * 
	 * @param xi
	 *                標本点集合の行列
	 * @param dst
	 *                重み和格納先
	 * 
	 */
	public void calcXByXi(TCMatrix xi, TCMatrix dst) {
		// wはベクトル
		dst.times(xi, fXW);
	}

	/**
	 * 標本点と中心のset1,2からcov(1,2)の共分散行列を求めてdstへセット
	 * 
	 * @param isQ1
	 *                1が状態次元かどうか
	 * @param xi1
	 *                1の標本点集合
	 * @param m1
	 *                1の中心
	 * @param isQ2
	 *                2が状態次元かどうか
	 * @param xi2
	 *                2の標本展集合
	 * @param m2
	 *                2の中心
	 * @param dst
	 *                共分散行列の格納先
	 */
	public void calcCovByXi(boolean isQ1, TCMatrix xi1, TCMatrix m1, boolean isQ2, TCMatrix xi2, TCMatrix m2, TCMatrix dst) {
		String key1;
		String key2;
		TCMatrix m1Tmp;
		TCMatrix m2Tmp;
		if (isQ1) {
			key1 = "QXi";
			m1Tmp = fTmp.get("Q1")[0].copyFrom(m1).times(-1.0);
		}
		else {
			key1 = "RXi";
			m1Tmp = fTmp.get("R1")[0].copyFrom(m1).times(-1.0);
		}
		if (isQ2) {
			key2 = "XiQ";
			m2Tmp = fTmp.get("1Q")[0].copyFrom(m2).times(-1.0);
		}
		else {
			key2 = "XiR";
			m2Tmp = fTmp.get("1R")[0].copyFrom(m2).times(-1.0);
		}
		TCMatrix delta1 = fTmp.get(key1)[0];
		TCMatrix tmpDst = fTmp.get(key1)[1];
		for (int i = 0; i < delta1.getColumnDimension(); i++) {
			delta1.copyAtColumn(m1Tmp, i);
		}
		delta1.add(xi1);
		TCMatrix delta2 = fTmp.get(key2)[0];
		TCMatrix txi2 = fTmp.get(key2)[1];
		txi2.tcopyFrom(xi2);
		for (int i = 0; i < delta2.getRowDimension(); i++) {
			delta2.copyAtRow(m2Tmp, i);
		}
		delta2.add(txi2);
		// fPWは対角行列で重み持つ
		tmpDst.times(delta1, fPW);
		dst.times(tmpDst, delta2);
	}

	@Override
	public void calcPByK() {
		TCMatrix[] qRTmp = fTmp.get("QR");
		TCMatrix[] rQTmp = fTmp.get("RQ");
		TCMatrix[] qqTmp = fTmp.get("QQ");
		qRTmp[0].times(fK, fZP);
		rQTmp[0].tcopyFrom(fK);
		qqTmp[0].times(qRTmp[0], rQTmp[0]);
		fFilterP.sub(fPredictP, qqTmp[0]);
	}

	@Override
	public void putLog() {
		super.putLog();
		Map<String, TCMatrix> entry = fHistry.get(fHistry.size() - 1);
		entry.put("fPredictXi", fPredictXi.clone());
		entry.put("fFilterXi", fFilterXi.clone());
		entry.put("fXW", fXW.clone());
		entry.put("fPW", fPW.clone());
	}

	/**
	 * 列ベクトルsrcの各次元に対応する各方程式を適用してdstへ格納 行列形式の方程式なら文字列のとこをnullにして呼ぶ(逆も)
	 * 
	 * @param src
	 *                引数となる列ベクトル
	 * @param dst
	 *                結果格納先
	 * @param fMatrix
	 *                行列形式
	 * @param fString
	 *                文字列形式
	 */
	public void applyFunction(TCMatrix src, TCMatrix dst, TCMatrix fMatrix, String[] fString) {
		if (fMatrix != null) {
			// 行列で更新する場合
			dst.times(fMatrix, src);
			return;
		}
		// 数式から更新する場合
		for (int i = 0; i < fVarList.length; i++) {
			// casに変数の値を代入
			String var = fVarList[i];
			if (var == null)
				continue;
			fUtil.evaluate(var + "=" + src.getValue(i));
		}
		for (int i = 0; i < fString.length; i++) {
			// casで式を評価して，dstに値をセットする
			double value = fUtil.evaluate(fString[i]).evalDouble();
			dst.setValue(i, 0, value);
		}

	}

	/**
	 * 標本点の重みを計算.具体クラスで実装
	 */
	public abstract void calcWeight();

}
