package filter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math4.util.Pair;

import matrix2017.TCMatrix;
import utility.TMyMatrixUtil;

public class TEnsembleKF extends TNonLinearKF {
	// 標本展の観測ノイズ
	TCMatrix fWNoise;

	public TEnsembleKF() {
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
	 * @param num
	 *                標本点の数
	 */
	public TEnsembleKF(TCMatrix initialX, TCMatrix initialP, TCMatrix Q, TCMatrix R, int num) {
		super(initialX, initialP, Q, R);
		fXiSize = num;
	}

	@Override
	public void initialize() {
		// 標本点をN(fX, fP)で初期化.fiterPyを参考
		super.initialize();
		fKeyList.add(new Pair<>("Xi", fXiSize));
		fWNoise = new TCMatrix(fR.getRowDimension(), fXiSize);
		addNoise(fFilterXi, fFilterP, fFilterX);
		calcWeight();
	}

	@Override
	public void calcWeight() {
		double wx = 1.0 / fXiSize;
		double wp = 1.0 / (fXiSize - 1);
		fXW = new TCMatrix(fXiSize, 1);
		fXW.fill(wx);
		fPW = new TCMatrix(fXiSize, fXiSize);
		for (int i = 0; i < fXiSize; i++) {
			fPW.setValue(i, i, wp);
		}
	}

	@Override
	public void predict() throws Exception {
		super.predict();
		// 内部ノイズを加える
		addNoise(fPredictXi, fQ, null);
		calcXByXi(fPredictXi, fPredictX);
		calcCovByXi(true, fPredictXi, fPredictX, true, fPredictXi, fPredictX, fPredictP);
	}

	@Override
	public void correct(TCMatrix z) {
		setSigmaWNoise();
		calcGamma();
		calcGain();
		// XiをFilterする(UKFと違うとこ)
		calcXiByK(z);
		calcXByXi(fFilterXi, fFilterX);
		calcCovByXi(true, fFilterXi, fFilterX, true, fFilterXi, fFilterX, fFilterP);
	}

	/**
	 * 標本点の観測誤差分散共分散であるfRHatを計算
	 */
	public void setSigmaWNoise() {
		fWNoise.fill(0.0);
		// シグマポイントのノイズを計算
		addNoise(fWNoise, fR, null);
		TCMatrix m = fTmp.get("RR")[0];
		m.fill(0.0);
		// fRHatを計算
		calcCovByXi(false, fWNoise, m, false, fWNoise, m, fRHat);
	}

	@Override
	public void readFrom(BufferedReader br) throws IOException {
		// TODO
		super.readFrom(br);
		fXiSize = Integer.parseInt(br.readLine());
	}

	/**
	 * 観測値zとカルマンゲインから標本点をフィルタリング
	 * 
	 * @param z
	 *                観測値
	 */
	public void calcXiByK(TCMatrix z) {
		// KからfilterXiを計算する
		TCMatrix[] rXi = fTmp.get("RXi");
		TCMatrix[] qXi = fTmp.get("QXi");
		for (int i = 0; i < rXi[0].getColumnDimension(); i++) {
			// rXi[0] <= [y,...]
			rXi[0].copyAtColumn(z, i);
		}
		// rx[0] <= w+y
		rXi[0].add(fWNoise);
		rXi[0].sub(fPredictGamma);
		qXi[0].times(fK, rXi[0]);
		fFilterXi.copyFrom(fPredictXi);
		fFilterXi.add(qXi[0]);
	}

	public static void main(String[] argv) throws Exception {
		// プロパティファイルから読み込みでもコンストラクタ呼び出しでも可
		TNonLinearKF kf = new TEnsembleKF();
		String property = kf.getDir() + "property.txt";
		BufferedReader br = new BufferedReader(new FileReader(property));
		kf.readFrom(br);
		// ログとるか.デフォルトだとfalse
		kf.setIsHistry(true);
		// 初期化
		kf.initialize();
		// テンポラリ行列セット
		kf.setTmp();
		br.close();
		BufferedReader zBr = new BufferedReader(new FileReader(kf.getDir() + "z_data.txt"));
		// 観測値の読み込み.
		ArrayList<TCMatrix> zList = new ArrayList<>();
		TCMatrix z = TMyMatrixUtil.readMatrix(zBr);
		while (z != null) {
			zList.add(z);
			z = TMyMatrixUtil.readMatrix(zBr);
		}
		zBr.close();
		for (TCMatrix tmp : zList) {
			// 観測値から予測とフィルタリング
			kf.calc(tmp);
			TCMatrix predicted = kf.fPredictX;
			TCMatrix filtered = kf.fFilterX;
			System.out.println(predicted);
			System.out.println(filtered);
		}
		// ログをcsvへ書き出す
		kf.outputLog();
	}
}
