interface User {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  profile?: UserProfile;
}

interface UserProfile {
  firstName: string;
  lastName: string;
  bio?: string;
  avatar?: string;
  location?: string;
}

interface AuthToken {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

class UserService {
  private users: Map<string, User> = new Map();
  private tokens: Map<string, AuthToken> = new Map();

  async register(
    username: string,
    email: string,
    password: string
  ): Promise<User> {
    // Validate input
    if (!this.isValidEmail(email)) {
      throw new Error('Invalid email format');
    }

    if (password.length < 8) {
      throw new Error('Password must be at least 8 characters');
    }

    // Check if user exists
    const existing = Array.from(this.users.values()).find(
      (u) => u.email === email || u.username === username
    );

    if (existing) {
      throw new Error('User already exists');
    }

    // Create user
    const user: User = {
      id: this.generateId(),
      username,
      email,
      createdAt: new Date(),
    };

    this.users.set(user.id, user);
    return user;
  }

  async login(
    username: string,
    password: string
  ): Promise<{ user: User; token: AuthToken }> {
    // Find user
    const user = Array.from(this.users.values()).find(
      (u) => u.username === username || u.email === username
    );

    if (!user) {
      throw new Error('Invalid credentials');
    }

    // Generate tokens
    const token: AuthToken = {
      accessToken: this.generateToken(),
      refreshToken: this.generateToken(),
      expiresIn: 3600,
    };

    this.tokens.set(user.id, token);

    return { user, token };
  }

  async getUserById(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async updateProfile(
    userId: string,
    profile: Partial<UserProfile>
  ): Promise<User> {
    const user = this.users.get(userId);

    if (!user) {
      throw new Error('User not found');
    }

    user.profile = {
      ...user.profile,
      ...profile,
    } as UserProfile;

    this.users.set(userId, user);
    return user;
  }

  async deleteUser(userId: string): Promise<void> {
    if (!this.users.has(userId)) {
      throw new Error('User not found');
    }

    this.users.delete(userId);
    this.tokens.delete(userId);
  }

  async verifyToken(token: string): Promise<User | null> {
    for (const [userId, authToken] of this.tokens.entries()) {
      if (authToken.accessToken === token) {
        return this.users.get(userId) || null;
      }
    }
    return null;
  }

  async refreshToken(refreshToken: string): Promise<AuthToken> {
    for (const [userId, authToken] of this.tokens.entries()) {
      if (authToken.refreshToken === refreshToken) {
        const newToken: AuthToken = {
          accessToken: this.generateToken(),
          refreshToken: this.generateToken(),
          expiresIn: 3600,
        };

        this.tokens.set(userId, newToken);
        return newToken;
      }
    }

    throw new Error('Invalid refresh token');
  }

  private isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  private generateToken(): string {
    return (
      Math.random().toString(36).substring(2) +
      Math.random().toString(36).substring(2)
    );
  }

  // Statistics and analytics
  async getUserStats(): Promise<{
    totalUsers: number;
    activeUsers: number;
    newUsersToday: number;
  }> {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

    const totalUsers = this.users.size;
    const activeUsers = this.tokens.size;
    const newUsersToday = Array.from(this.users.values()).filter(
      (u) => u.createdAt >= today
    ).length;

    return {
      totalUsers,
      activeUsers,
      newUsersToday,
    };
  }
}

// Export singleton instance
export const userService = new UserService();

// Example usage
async function example() {
  try {
    // Register new user
    const user = await userService.register(
      'john_doe',
      'john@example.com',
      'secure_password123'
    );
    console.log('User registered:', user);

    // Login
    const { user: loggedInUser, token } = await userService.login(
      'john_doe',
      'secure_password123'
    );
    console.log('Login successful:', loggedInUser);
    console.log('Access token:', token.accessToken);

    // Update profile
    await userService.updateProfile(user.id, {
      firstName: 'John',
      lastName: 'Doe',
      bio: 'Software developer',
    });

    // Get stats
    const stats = await userService.getUserStats();
    console.log('User statistics:', stats);
  } catch (error) {
    console.error('Error:', error);
  }
}
