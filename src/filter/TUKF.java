package filter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math4.util.Pair;

import matrix2017.TCMatrix;
import matrix2017.decompositions.TCCholeskyDecomposition;
import utility.TMyMatrixUtil;

public class TUKF extends TNonLinearKF {
	// シグマポイントのパラメータ推奨値
	double fAlpha = 0.1;// 1e-3説もある
	double fBeta = 2;
	double fKappa = -1;
	double fLambda;
	double fL;

	public TUKF() {
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
	public TUKF(TCMatrix initialX, TCMatrix initialP, TCMatrix Q, TCMatrix R) {
		super(initialX, initialP, Q, R);
		setXiSize();
	}

	/**
	 * シグマポイントの個数をシステムノイズ行列の次元からセット
	 */
	public void setXiSize() {
		fXiSize = 2 * fQ.getRowDimension() + 1;
	}

	@Override
	public void readFrom(BufferedReader br) throws IOException {
		super.readFrom(br);
		setXiSize();
	}

	@Override
	public void calcWeight() {
		fL = fQ.getRowDimension();
		fXiSize = (int) fL * 2 + 1;
		fLambda = fAlpha * fAlpha * (fL + fKappa) - fL;
		fXW = new TCMatrix(fXiSize, 1);
		fPW = new TCMatrix(fXiSize, fXiSize);
		double xw0 = fLambda / (fL + fLambda);
		double pw0 = fLambda / (fL + fLambda) + 1.0 - fAlpha * fAlpha + fBeta;
		double w1 = 0.5 / (fL + fLambda);
		fXW.fill(w1);
		fXW.setValue(0, xw0);
		fPW.setValue(0, 0, pw0);
		// fPWは対角行列
		for (int i = 1; i < fXiSize; i++) {
			fPW.setValue(i, i, w1);
		}
	}

	/**
	 * 
	 * @param m
	 *                中心ベクトル
	 * @param p
	 *                分散共分散行列
	 * @param dst
	 *                シグマポイントの行列.(次元,シグマポイントの個数)の形式
	 */
	public void calcXi(TCMatrix m, TCMatrix p, TCMatrix dst) {
		TCMatrix[] tmp = fTmp.get("QQ");
		tmp[0].copyFrom(p);
		TCCholeskyDecomposition chol = tmp[0].times(fLambda + fL).chol();
		// choleskyはnew強制.
		TCMatrix rootP = chol.getL();
		// 中心ベクトルは先頭
		dst.copyAtColumn(m, 0);
		tmp = fTmp.get("Q1");
		for (int i = 0; i < (int) fL; i++) {
			// 1 ~ Lのシグマポイント
			TCMatrix vec = tmp[0];
			vec.copySubmatrixFrom(rootP, 0, rootP.getRowDimension() - 1, i, i, 0, 0);
			vec.add(m);
			dst.copyAtColumn(vec, i + 1);
		}
		rootP.times(-1.0);
		int start = (int) fL;
		for (int i = 0; i < (int) fL; i++) {
			// L+1 ~ 2Lのシグマポイント
			TCMatrix vec = tmp[0];
			vec.copySubmatrixFrom(rootP, 0, rootP.getRowDimension() - 1, i, i, 0, 0);
			vec.add(m);
			dst.copyAtColumn(vec, start + i + 1);
		}
	}

	@Override
	public void predict() throws Exception {
		calcXi(fFilterX, fFilterP, fFilterXi);
		super.predict();
		calcXByXi(fPredictXi, fPredictX);
		calcCovByXi(true, fPredictXi, fPredictX, true, fPredictXi, fPredictX, fPredictP);
		fPredictP.add(fQ);
	}

	@Override
	public void correct(TCMatrix z) throws Exception {
		// filterPyではやってない
		// calcXi(fPredictX, fPredictP, fPredictXi); // 標本の再計算
		super.correct(z);
		calcXi(fFilterX, fFilterP, fFilterXi);
		// 記録するとき用.予測・ろはだけなら不要
	}

	@Override
	public void initialize() {
		super.initialize();
		setXiSize();
		fKeyList.add(new Pair<>("Xi", fXiSize));
	}

	public static void main(String[] argv) throws Exception {
		// プロパティファイルからでもコンストラクタ呼び出しでも可
		TNonLinearKF kf = new TUKF();
		String property = kf.getDir() + "property.txt";
		BufferedReader br = new BufferedReader(new FileReader(property));
		kf.readFrom(br);
		// ログとるなら.デフォルトではfalse
		kf.setIsHistry(true);
		// 初期化
		kf.initialize();
		// テンポラリ行列セット
		kf.setTmp();
		br.close();
		BufferedReader zBr = new BufferedReader(new FileReader(kf.getDir() + "z_data.txt"));
		// 観測値を読み出し.読み出さないで作ってもよい
		ArrayList<TCMatrix> zList = new ArrayList<>();
		TCMatrix z = TMyMatrixUtil.readMatrix(zBr);
		while (z != null) {
			zList.add(z);
			z = TMyMatrixUtil.readMatrix(zBr);
		}
		zBr.close();
		for (TCMatrix tmp : zList) {
			kf.calc(tmp);
			TCMatrix predicted = kf.fPredictX;
			TCMatrix filtered = kf.fFilterX;
			System.out.println(predicted);
			System.out.println(filtered);
		}
		// ロき書き出し
		kf.outputLog();
	}
}
