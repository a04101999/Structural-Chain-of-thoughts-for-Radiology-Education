import React from "react";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { SurveySettings } from "@/types/study";

type CheckedState = boolean | "indeterminate";

type SettingsDialogProps = {
	surveySettings: {
		displayPointRadius: number;
		pointDistance: number;
		baseDisplayPointDuration: number;
		speedUpDisplayPointDuration: number;
		showGroundTruthPoints: CheckedState;
	};
	setSurveySettings: React.Dispatch<React.SetStateAction<SurveySettings>>;
	handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
};

const SettingsDialog: React.FC<SettingsDialogProps> = ({
	surveySettings,
	setSurveySettings,
	handleSubmit,
}) => {
	return (
		<Dialog>
			<DialogTrigger asChild>
				<Button
					className="hidden absolute top-4 right-4 border bg-neutral-100 border-gray-300 w-24 shadow-sm rounded-md px-3 py-2 z-20 opacity-25 hover:opacity-100"
					variant="outline"
				>
					Settings
				</Button>
			</DialogTrigger>
			<DialogContent className="sm:max-w-md">
				<DialogHeader>
					<DialogTitle>Edit settings</DialogTitle>
					<DialogDescription className="text-gray-600">
						Edit the settings for the survey.
					</DialogDescription>
				</DialogHeader>
				<form onSubmit={handleSubmit} className="flex flex-col gap-4">
					<div>
						<Label htmlFor="displayPointRadius" className="pb-1">
							Point Radius (px)
						</Label>
						<Input
							id="displayPointRadius"
							name="displayPointRadius"
							value={surveySettings.displayPointRadius}
							onChange={(e) =>
								setSurveySettings((prev) => ({
									...prev,
									displayPointRadius: parseInt(e.target.value),
								}))
							}
							className="w-full border-gray-400"
							type="number"
							min={1}
							max={100}
							step={1}
							required
						/>
					</div>

					<div>
						<Label htmlFor="pointDistance" className="pb-1">
							Point Distance (px)
						</Label>
						<Input
							id="pointDistance"
							name="pointDistance"
							value={surveySettings.pointDistance}
							onChange={(e) =>
								setSurveySettings((prev) => ({
									...prev,
									pointDistance: parseInt(e.target.value),
								}))
							}
							className="w-full border-gray-400"
							type="number"
							min={1}
							max={100}
							step={1}
							required
						/>
					</div>

					<div>
						<Label htmlFor="baseDisplayPointDuration" className="pb-1">
							Base Point Display Duration (ms)
						</Label>
						<Input
							id="baseDisplayPointDuration"
							name="baseDisplayPointDuration"
							value={surveySettings.baseDisplayPointDuration}
							onChange={(e) =>
								setSurveySettings((prev) => ({
									...prev,
									baseDisplayPointDuration: parseInt(e.target.value),
								}))
							}
							className="w-full border-gray-400"
							type="number"
							min={1}
							max={100000}
							step={1}
							required
						/>
					</div>

					<div>
						<Label htmlFor="speedUpDisplayPointDuration" className="pb-1">
							Accelerated Point Display Duration (ms)
						</Label>
						<Input
							id="speedUpDisplayPointDuration"
							name="speedUpDisplayPointDuration"
							value={surveySettings.speedUpDisplayPointDuration}
							onChange={(e) =>
								setSurveySettings((prev) => ({
									...prev,
									speedUpDisplayPointDuration: parseInt(e.target.value),
								}))
							}
							className="w-full border-gray-400"
							type="number"
							min={1}
							max={100000}
							step={1}
							required
						/>
					</div>

					<div className="flex items-center gap-2">
						<Checkbox
							name="showGroundTruthPoints"
							checked={surveySettings.showGroundTruthPoints}
							onCheckedChange={(checked) => {
								setSurveySettings((prev) => ({
									...prev,
									showGroundTruthPoints: checked,
								}));
							}}
							id="showGroundTruthPoints"
						/>
						<label
							htmlFor="showGroundTruthPoints"
							className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
						>
							Show ground truth points
						</label>
					</div>

					<DialogFooter>
						<Button type="submit">Save</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
};

export default SettingsDialog;
