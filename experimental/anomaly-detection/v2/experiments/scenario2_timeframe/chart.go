package main

import (
	"fmt"
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func writeChart(path string, results []result) error {
	p := plot.New()
	p.X.Label.Text = "Training Duration (Weeks)"
	p.Y.Label.Text = "Metric Value (%)"
	grid := plotter.NewGrid()
	grid.Vertical.Color = nil
	grid.Horizontal.Color = color.Gray{Y: 230}
	p.Add(grid)
	p.X.Label.TextStyle.Font.Variant, p.Y.Label.TextStyle.Font.Variant = "Sans", "Sans"
	p.X.Tick.Label.Font.Variant, p.Y.Tick.Label.Font.Variant = "Sans", "Sans"
	p.Legend.TextStyle.Font.Variant = "Sans"
	series := []struct {
		name  string
		color color.RGBA
		shape draw.GlyphDrawer
		value func(metricSet) float64
	}{
		{"Accuracy", color.RGBA{26, 115, 232, 255}, draw.CircleGlyph{}, func(m metricSet) float64 { return m.accuracy }},
		{"Precision", color.RGBA{52, 168, 83, 255}, draw.BoxGlyph{}, func(m metricSet) float64 { return m.precision }},
		{"Recall", color.RGBA{255, 171, 0, 255}, draw.PyramidGlyph{}, func(m metricSet) float64 { return m.recall }},
	}
	for _, s := range series {
		points := make(plotter.XYs, len(results))
		for i, r := range results {
			points[i].X, points[i].Y = float64(r.weeks), 100*s.value(r.metrics)
		}
		line, dots, err := plotter.NewLinePoints(points)
		if err != nil {
			return err
		}
		line.Color, line.Width = s.color, vg.Points(2)
		dots.Color, dots.Shape, dots.Radius = s.color, s.shape, vg.Points(3)
		p.Add(line, dots)
		p.Legend.Add(s.name, line, dots)
	}
	p.X.Min, p.X.Max, p.Y.Min, p.Y.Max = 0, 51, 0, 105
	var ticks []plot.Tick
	for _, weeks := range []int{1, 4, 8, 12, 16, 20, 30, 40, 50} {
		ticks = append(ticks, plot.Tick{Value: float64(weeks), Label: fmt.Sprint(weeks)})
	}
	p.X.Tick.Marker = plot.ConstantTicks(ticks)
	var percentages []plot.Tick
	for value := 0; value <= 100; value += 20 {
		percentages = append(percentages, plot.Tick{Value: float64(value), Label: fmt.Sprint(value)})
	}
	p.Y.Tick.Marker = plot.ConstantTicks(percentages)
	p.Legend.Top, p.Legend.Left = false, false
	p.Legend.XOffs, p.Legend.YOffs, p.Legend.Padding = -vg.Points(12), vg.Points(12), vg.Points(4)
	return p.Save(10*vg.Inch, 4.5*vg.Inch, path)
}
