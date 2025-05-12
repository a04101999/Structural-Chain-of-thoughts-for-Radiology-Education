// import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import ErrorPage from "./routes/ErrorPage";
import { Toaster } from "sonner";
import Study from "./routes/Study";
import Validation from "./routes/Validation";

const router = createBrowserRouter([
	{
		path: "/",
		element: <Validation />,
		errorElement: <ErrorPage />,
	},
	{
		path: "/study",
		element: <Study />,
	},
]);

createRoot(document.getElementById("root")!).render(
	<>
		<RouterProvider router={router} />
		<Toaster
			position="top-center"
			toastOptions={{ className: "text-base font-medium" }}
			richColors
			visibleToasts={2}
			duration={2500}
		/>
	</>
);
