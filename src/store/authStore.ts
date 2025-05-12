import { create } from "zustand";

interface Credentials {
	firstName: string;
	lastName: string;
	email: string;
	isAuthenticated: boolean;
}

interface AuthState {
	credentials: Credentials;
	setState: (creds: Credentials) => void;
	reset: () => void;
}

export const useAuthStore = create<AuthState>()((set) => ({
	credentials: {
		firstName: "",
		lastName: "",
		email: "",
		isAuthenticated: false,
	},
	setState: (creds: Credentials) => set({ credentials: creds }),
	reset: () =>
		set({
			credentials: {
				firstName: "",
				lastName: "",
				email: "",
				isAuthenticated: false,
			},
		}),
}));
