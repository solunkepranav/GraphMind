class AccessControlManager:
    """Source-level Access Control List (ACL) manager.
    Enforces the security invariant: effective_sources = ACL_allowed ∩ user_selected.
    """

    def __init__(self, role_permissions: dict[str, list[str]] = None):
        # Default role mapping
        self.role_permissions = role_permissions or {
            "admin": ["*"],
            "engineering": ["architecture.md", "spec.md", "README.md", "code.py"],
            "finance": ["financials.pdf", "revenue_q3.pdf", "report.pdf"],
            "general": ["public.pdf", "handbook.pdf"]
        }

    def get_allowed_sources(self, role: str, all_sources: list[str]) -> list[str]:
        """Returns list of sources permitted for given role."""
        permitted = self.role_permissions.get(role, [])
        if "*" in permitted:
            return list(all_sources)
        return [s for s in all_sources if s in permitted]

    def get_effective_sources(
        self,
        role: str,
        all_sources: list[str],
        user_selected_sources: list[str] = None
    ) -> list[str]:
        """Computes effective sources as: ACL_allowed ∩ user_selected.
        Guarantees that a user cannot access documents outside their role permissions.
        """
        allowed = set(self.get_allowed_sources(role, all_sources))
        if user_selected_sources is None:
            return list(allowed)
        selected = set(user_selected_sources)
        return list(allowed.intersection(selected))
