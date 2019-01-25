using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;


namespace gtest1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = this.CreateGraphics();

            g.Clear(Color.White);
            //Math.Cos();

            int width = this.Width;
            int height = this.Height;

            PointF centerOfEllipse = new PointF(width/2, height/2);
            SizeF sizeOfEllipse = new SizeF(100, 100);
            PointF leftUpCornerOfEllipse = new PointF(centerOfEllipse.X - sizeOfEllipse.Width/2, centerOfEllipse.Y - sizeOfEllipse.Height/2);            
            RectangleF ellipseRect = new RectangleF(leftUpCornerOfEllipse, sizeOfEllipse);

            g.DrawEllipse(new Pen(Color.Red), ellipseRect);

            int nMaxLines = 300;
            
            for (int i = 0; i < nMaxLines; ++i)
            {
                float angle = (1.0f/nMaxLines * i) * (2.0f * (float)Math.PI);
                float scale = 400.0f;
                PointF direction = new PointF((float)Math.Cos(angle), (float)Math.Sin(angle));
                float initialOffset = 60;
                PointF lineStart = new PointF(direction.X * initialOffset + centerOfEllipse.X, direction.Y * initialOffset + centerOfEllipse.Y);
                g.DrawLine(new Pen(Color.Blue), lineStart, new PointF(direction.X * scale + centerOfEllipse.X, direction.Y * scale + centerOfEllipse.Y));
            }

            SizeF sizeOfEllipse2 = new SizeF(800, 800);
            PointF leftUpCornerOfEllipse2 = new PointF(centerOfEllipse.X - sizeOfEllipse2.Width / 2, centerOfEllipse.Y - sizeOfEllipse2.Height / 2);            
            g.DrawEllipse(new Pen(Color.Red), new RectangleF(leftUpCornerOfEllipse2, sizeOfEllipse2));

            g.Dispose();
        }
    }
}
