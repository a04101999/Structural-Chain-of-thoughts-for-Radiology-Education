import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

export default function ErrorPage() {
	const navigate = useNavigate();
	return (
		<div className="w-full h-screen flex items-center justify-center">
			<div className="flex flex-col items-center">
				<p className="text-6xl font-bold text-gray-800 mb-1">404</p>
				<p className="text-2xl font-bold text-gray-800 mb-4">PAGE NOT FOUND</p>
				<Button
					onClick={() => {
						navigate("/");
					}}
					className="text-base font-medium px-4 py-5"
				>
					Take me home
				</Button>
			</div>
		</div>
	);
}
