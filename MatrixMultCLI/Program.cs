using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixMultCLI
{
    class Program
    {
        static void Main(string[] args)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            double[][] ma = MatrixCreate(512, 512);
            double[][] mb = MatrixCreate(512, 512);

            for(int i = 0; i < 512; ++i)
            {
                for (int j = 0; j < 512; ++j)
                {
                    ma[i][j] = (i * 512) + (j + 1);
                    mb[i][j] = (i * 512) + (j + 1);
                }
            }
            MatrixProduct(ma, mb);
            watch.Stop();
            Console.WriteLine("Sequential matrix multiplication took: " + watch.ElapsedMilliseconds + "ms");

            watch.Reset();
            watch.Start();
            ma = MatrixCreate(512, 512);
            mb = MatrixCreate(512, 512);

            Parallel.For(0, 512, i =>
            { 
                for(int j = 0; j < 512; ++j)
                {
                    ma[i][j] = (i*512) + (j + 1);
                    mb[i][j] = (i*512) + (j + 1);
                }
            });
            MatrixParallelProduct(ma, mb);
            watch.Stop();
            Console.WriteLine("Parallel matrix multiplication took: " + watch.ElapsedMilliseconds + "ms");
        }

        static double[][] MatrixCreate(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
            {
                result[i] = new double[cols];
            }
            return result;
        }

        static double[][] MatrixProduct(double[][] matrixA, double[][] matrixB)
        {
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;

            double[][] result = MatrixCreate(aRows, bCols);

            for (int i = 0; i < aRows; ++i)
            {
                for (int j = 0; j < bCols; ++j)
                {
                    for (int k = 0; k < aCols; ++k)
                    {
                        result[i][j] += matrixA[i][k] * matrixB[k][j];
                    }
                }
            }

            return result;
        }

        static double[][] MatrixParallelProduct(double[][] matrixA, double[][] matrixB)
        {
            int aRows = matrixA.Length; int aCols = matrixA[0].Length;
            int bRows = matrixB.Length; int bCols = matrixB[0].Length;

            double[][] result = MatrixCreate(aRows, bCols);

            Parallel.For(0, aRows, i =>
            {
                for(int j = 0; j < bCols; ++j)
                {
                    for(int k = 0; k < aCols; ++k)
                    {
                        result[i][j] += matrixA[i][k] * matrixB[k][j];
                    }
                }
            });

            return result;
        }
    }
}
