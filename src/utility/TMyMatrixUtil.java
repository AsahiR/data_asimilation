package utility;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import matrix2017.TCMatrix;

public class TMyMatrixUtil {

	/**
	 * 
	 * @param m
	 * @return (row,column)\n 成分をcsvにして文字列
	 */
	public static String logMatrix(TCMatrix m) {
		String str = m.getRowDimension() + "," + m.getColumnDimension() + "\n";
		for (int r = 0; r < m.getRowDimension(); r++) {
			for (int c = 0; c < m.getColumnDimension(); c++) {
				double val = m.getValue(r, c);
				if (c > 0)
					str += ",";
				str += val;
			}
			str += "\n";
		}
		return str;
	}

	public static void readMatrix(TCMatrix matrix, String src) {
		try (BufferedReader br = new BufferedReader(new FileReader(src))) {
			int row = 0;
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				String[] splits = line.split(",");
				for (int c = 0; c < splits.length; c++) {
					double val = Double.parseDouble(splits[c]);
					matrix.setValue(row, c, val);
				}
				row++;
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

	public static TCMatrix readMatrix(BufferedReader br) throws IOException {
		String line = br.readLine();
		if (line == null) {
			return null;
		}
		String[] splits = line.split(",");
		int rowSize = Integer.parseInt(splits[0]);
		int columnSize = Integer.parseInt(splits[1]);
		TCMatrix mat = new TCMatrix(rowSize, columnSize);
		for (int row = 0; row < rowSize; row++) {
			line = br.readLine();
			splits = line.split(",");
			for (int column = 0; column < columnSize; column++) {
				double val = Double.parseDouble(splits[column]);
				mat.setValue(row, column, val);
			}
		}
		return mat;
	}

}
