import { useRef, useEffect } from "react";

type Point = {
	x: number;
	y: number;
};

function euclideanDistance(pt1: Point, pt2: Point) {
	return Math.sqrt(Math.pow(pt1.x - pt2.x, 2) + Math.pow(pt1.y - pt2.y, 2));
}

function linearInterpolation(x: number, point1: Point, point2: Point) {
	return (
		point1.y + (x - point1.x) * ((point2.y - point1.y) / (point2.x - point1.x))
	);
}

export default function Test() {
	const canvasRef = useRef<HTMLCanvasElement>(null);

	// https://processing.org/reference/lerp_.html

	useEffect(() => {
		const drawPoint = () => {
			const ctx = canvasRef.current?.getContext("2d");

			if (!ctx) {
				return;
			}

			ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
			ctx.fillStyle = "black";
			const pt1X = 150;
			const pt1Y = Math.floor(Math.random() * ctx.canvas.height);
			// const pt1X = Math.floor(Math.random() * ctx.canvas.width);
			// const pt1Y = Math.floor(Math.random() * ctx.canvas.height);
			const circle = new Path2D();
			circle.arc(pt1X, pt1Y, 12, 0, 2 * Math.PI);
			ctx.fill(circle);

			const pt2X = 150;
			const pt2Y = Math.floor(Math.random() * ctx.canvas.height);
			// const pt2X = Math.floor(Math.random() * ctx.canvas.width);
			// const pt2Y = Math.floor(Math.random() * ctx.canvas.height);
			const circle2 = new Path2D();
			circle2.arc(pt2X, pt2Y, 12, 0, 2 * Math.PI);
			ctx.fillStyle = "red";
			ctx.fill(circle2);

			console.log(
				"pt1X: ",
				pt1X,
				"pt1Y: ",
				pt1Y,
				"pt2X: ",
				pt2X,
				"pt2Y: ",
				pt2Y
			);
			const slope = (pt2Y - pt1Y) / (pt2X - pt1X);
			const distance = euclideanDistance(
				{ x: pt1X, y: pt1Y },
				{ x: pt2X, y: pt2Y }
			);
			console.log(slope);
			console.log(distance);

			const interpolatedPoints: Point[] = [];

			const pointDistance = 20;
			const radius = 12;

			if (pt1X < pt2X) {
				for (
					let x = pt1X + pointDistance + radius;
					x < pt2X;
					x += pointDistance + radius
				) {
					const y = linearInterpolation(
						x,
						{ x: pt1X, y: pt1Y },
						{ x: pt2X, y: pt2Y }
					);
					const circle = new Path2D();
					circle.arc(x, y, radius, 0, 2 * Math.PI);
					ctx.fillStyle = "green";
					ctx.fill(circle);

					interpolatedPoints.push({ x, y });
				}
			} else if (pt1X > pt2X) {
				for (
					let x = pt1X - pointDistance - radius;
					x > pt2X;
					x -= pointDistance + radius
				) {
					const y = linearInterpolation(
						x,
						{ x: pt1X, y: pt1Y },
						{ x: pt2X, y: pt2Y }
					);
					const circle = new Path2D();
					circle.arc(x, y, radius, 0, 2 * Math.PI);
					ctx.fillStyle = "green";
					ctx.fill(circle);

					interpolatedPoints.push({ x, y });
				}
			} else if (pt1Y < pt2Y) {
				for (
					let y = pt1Y + pointDistance + radius;
					y < pt2Y;
					y += pointDistance + radius
				) {
					const circle = new Path2D();
					circle.arc(pt2X, y, radius, 0, 2 * Math.PI);
					ctx.fillStyle = "green";
					ctx.fill(circle);

					interpolatedPoints.push({ x: pt2X, y });
				}
			} else if (pt1Y > pt2Y) {
				for (
					let y = pt1Y - pointDistance - radius;
					y > pt2Y;
					y -= pointDistance + radius
				) {
					const circle = new Path2D();
					circle.arc(pt2X, y, radius, 0, 2 * Math.PI);
					ctx.fillStyle = "green";
					ctx.fill(circle);

					interpolatedPoints.push({ x: pt2X, y });
				}
			}

			console.log(interpolatedPoints);

			ctx.beginPath();
			ctx.moveTo(pt1X, pt1Y);
			ctx.lineTo(pt2X, pt2Y);
			ctx.strokeStyle = "blue";
			ctx.lineWidth = 2;
			ctx.stroke();
		};

		drawPoint();
	}, []);

	return (
		<div className="h-screen w-full">
			<canvas
				ref={canvasRef}
				id="canvas"
				height={window.innerHeight}
				width={window.innerWidth}
			></canvas>
		</div>
	);
}
